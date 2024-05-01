// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <regex> 
#include <fstream>

std::string generatePreSystemPrompt(const std::string& promptFormat);
std::string generatePostSystemPrompt(const std::string& promptFormat);
std::string generatePreAnswer(const std::string& promptFormat);
std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat);
std::string createResponsePrompt(const std::string& systemPrompt, const std::string& modelThoughts, const std::string& promptFormat);
bool containsYesOrNo(const std::string& input);
static std::vector<float> softmax(const std::vector<float>& logits);
std::string quoteAndEscape(const std::string& input, int maxLen);

// Struct which will serve as the value of a key-value pair in an unordered map.
struct info {
    std::string inputText;
    std::string inputDate;
    std::string output;
    double yesProb;
    double noProb;
};

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

static std::vector<std::string> k_prompts;

static std::string default_system =
R"(The text provided is a pathology report, with samples originating from the colon or rectum unless specified otherwise. 
We are interested in identifying whether invasive adenocarcinoma is present in *any* colon or rectal sample. 
Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. 
Does the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?)";

std::string generatePreSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return " [INST] ";
    } else if (promptFormat == "llama3") {
        return "<|start_header_id|>system<|end_header_id|>\n\n";
    } else if (promptFormat == "phi3") {
        return "<user>\n ";
    //} else if (promptFormat == "phi3") {
    //    return "<system>\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePostSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return "\n<<<\nPathology report: ";
    } else if (promptFormat == "llama3") {
        return "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPathology report: '";
    } else if (promptFormat == "phi3") {
        return "\n<<<\nPathology report: '";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePreAnswer(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return "\n>>>\nDoes the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample? [/INST] Answer:";
    } else if (promptFormat == "llama3") {
        return "'\n\nDoes the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nAnswer:";
    } else if (promptFormat == "phi3") {
        return "\n>>>\nDoes the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample? <|end|>\n <|assistant|> Answer:";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat) {
    std::string preSystem = generatePreSystemPrompt(promptFormat);
    std::string postSystem = generatePostSystemPrompt(promptFormat);
    std::string preAnswer = generatePreAnswer(promptFormat);
    
    return preSystem + systemPrompt + postSystem;
}

struct client {
    ~client() {
        if (ctx_sampling) {
            llama_sampling_free(ctx_sampling);
        }
    }

    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    struct llama_sampling_context * ctx_sampling = nullptr;
};

static void print_date_time() {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    printf("\n\033[35mrun parameters as at %s\033[0m\n", buffer);
}

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Define a function to evaluate yes and no
bool containsYesOrNo(const std::string& input) {
    // Convert the input string to lowercase for case-insensitive comparison
    std::string lowerInput = input;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // Define regex patterns to check for standalone "yes" or "no"
    std::regex yesNoPattern("\\b(yes|no)\\b"); // Word boundary, "yes" or "no", word boundary

    // Use regex_search to find the pattern in the string
    return std::regex_search(lowerInput, yesNoPattern);
}

// Define softmax function from examples/perplexity
static std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

// Function to escape quotes, newlines, and tabs, and enclose the first 20 characters of the string in quotes
std::string quoteAndEscape(const std::string& input, int maxLen) {
    std::string truncated;
    if (maxLen == -1) {
        truncated = input;  // Use the full string if maxLen is -1
    } else {
        truncated = input.substr(0, maxLen); // Truncate to maxLen characters
    }

    std::string output = "\"";  // Start with an opening quote
    for (char ch : truncated) {
        switch (ch) {
            case '"':
                output += "\"\"";  // Escape quotes by doubling them
                break;
            case '\n':
                output += "\\n";   // Escape newlines
                break;
            case '\t':
                output += "\\t";   // Escape tabs
                break;
            default:
                output += ch;
        }
    }
    output += "\"";  // End with a closing quote
    return output;
}

size_t promptNumber = 0;

int main(int argc, char ** argv) {
    srand(1234);

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    // dedicate one sequence to the system prompt
    params.n_parallel += 1;

    // requests to simulate
    const int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;

    const bool dump_kv_cache = params.dump_kv_cache;

    // Get current time
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);
    // Buffer to hold the date-time format
    char dateTimeBuffer[20];  // Ensure the buffer is large enough for the format
    // Format the date and time with strftime
    strftime(dateTimeBuffer, sizeof(dateTimeBuffer), "%H%M_%d%m%Y", now_tm);
    // Convert to string for use in filenames or other outputs
    std::string dateTimeOutFile = dateTimeBuffer;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("parallel", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the target model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    // Set file names
    std::string dirPath = params.outDir;
    std::string inputFile = dirPath + "/inputTextNoFormatting_" + dateTimeOutFile + ".txt";
    std::string systemFile = dirPath + "/systemPromptNoFormatting_" + dateTimeOutFile + ".txt";
    std::string outputFile = dirPath + "/outputYN_withProbs_" + dateTimeOutFile + ".txt";
    std::string metadataFile = dirPath + "/metadata_" + dateTimeOutFile + ".txt";

    std::vector<std::string> prompts;
    // load the prompts from an external file if there are any
    if (params.prompt.empty()) {
        throw std::runtime_error("Error: No prompts given");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        printf("\n\033[32mNow printing the external prompt file %s\033[0m\n\n", params.prompt_file.c_str());


        // Create and open a text file
        std::ofstream outFile1(inputFile.c_str());

        // Check if the file was opened successfully
        if (!outFile1) {
            std::cerr << "Failed to open the input prompt out file." << std::endl;
            return 1; // Return with error code
        }

        prompts = split_string(params.prompt, '\n');
        std::string tmpPrompt;
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            tmpPrompt = prompt + generatePreAnswer(params.promptFormat);
            k_prompts[index] = tmpPrompt;
            index++;
            printf("%3d prompt: %s\n", index, tmpPrompt.c_str());

            // Write each prompt to the out file
            outFile1 << prompt << std::endl; // Adding newline for separation in file
        }

        // Close the file
        outFile1.close();
        
    }

    fprintf(stderr, "\n\n");
    fflush(stderr);

    const int n_ctx = llama_n_ctx(ctx);

    // Write format to the metadataFile
    std::string promptFormat_example = formatSystemPrompt("{System prompt here}", params.promptFormat) + "{Input text here}" + generatePreAnswer(params.promptFormat);
    std::vector<llama_token> tokens_format;
    // Bool in third arg represents BOS token, which we DO want here.
    tokens_format = ::llama_tokenize(ctx, promptFormat_example, true);
    // Create and open a text file to save the promptFormat
    std::ofstream outFile2(metadataFile.c_str());

    // Check if the file was opened successfully
    if (!outFile2) {
        std::cerr << "Failed to open the metadata out file." << std::endl;
        return 1; // Return with error code
    }

    // Write each prompt to the out file
    outFile2 << "Output file format: {Y/N text}\\t{Yes Prob.}\\t{No Prob.}\\t\"{First 20 chars of input (to make sure we have the right input mapped to the right output. \\n's and \\t's are escaped)}\"" << std::endl << std::endl;
    outFile2 << "Model path: " << params.model << std::endl << std::endl;
    outFile2 << quoteAndEscape(promptFormat_example, -1) << std::endl << std::endl << "Prompt format tokenized:" << std::endl; // Adding newline for separation in file

    // Iterate through the vector and write each element to the file
    for (size_t i = 0; i < tokens_format.size(); ++i) {
        outFile2 << tokens_format[i] << std::endl;
    }

    // Close the file
    outFile2.close();

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        //// Modify to selectively use grammar. Probably don't want it here but want it later.
        client.ctx_sampling = llama_sampling_init(params.sparams);
    }

    std::vector<llama_token> tokens_system;
    std::string system = default_system;
    if(!params.systemPrompt.empty()){
        system = params.systemPrompt;
    }


    // Create and open a text file to save the system prompt
    std::ofstream outFile3(systemFile.c_str());

    // Check if the file was opened successfully
    if (!outFile3) {
        std::cerr << "Failed to open the system prompt out file." << std::endl;
        return 1; // Return with error code
    }

    // Write each prompt to the out file
    outFile3 << system << std::endl; // Adding newline for separation in file

    // Close the file
    outFile3.close();

    std::string k_system = formatSystemPrompt(system, params.promptFormat);
    // Print the string
    printf("System prompt: %s\n", k_system.c_str());
    tokens_system = ::llama_tokenize(ctx, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();

    llama_seq_id g_seq_id = 0;

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);

    const auto t_main_start = ggml_time_us();

    LOG_TEE("%s: Simulating parallel requests from clients:\n", __func__);
    LOG_TEE("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d, prompt format =%s\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system, params.promptFormat.c_str());
    LOG_TEE("\n");

    {
        LOG_TEE("%s: Evaluating the system prompt ...\n", __func__);

        for (int32_t i = 0; i < n_tokens_system; ++i) {
            llama_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i <= n_clients; ++i) {
            // Copying the cache from client 0 to all n_clients clients (what are the last two args?)
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }

        LOG_TEE("\n");
    }

    LOG_TEE("Processing requests ...\n\n");

    // Open output file to write to
    std::ofstream outFile4(outputFile.c_str());
    // Check if the file was opened successfully
    if (!outFile4) {
        std::cerr << "Failed to open the output out file." << std::endl;
        return 1; // Return with error code
    }

    while (true) {
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            dump_kv_cache_view_seqs(kvc_view, 40);
        }

        llama_batch_clear(batch);

        // decode any currently ongoing sequences
        for (auto & client : clients) {
            if (client.seq_id == -1) {
                continue;
            }

            client.i_batch = batch.n_tokens;

            llama_batch_add(batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, { client.id + 1 }, true);

            client.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 1; i <= n_clients; ++i) {
                llama_kv_cache_seq_rm(ctx, i, -1, -1);
                // but keep the system prompt
                llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
            }

            LOG_TEE("%s: clearing the KV cache\n", __func__);
        }

        // insert new sequences for decoding
        if (cont_batching || batch.n_tokens == 0) {
            for (auto & client : clients) {
                if (client.seq_id == -1 && g_seq_id < n_seq) {
                    client.seq_id = g_seq_id;

                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    client.input    = k_prompts[promptNumber];
                    promptNumber++;
                    client.prompt   = client.input;
                    client.response = "";

                    llama_sampling_reset(client.ctx_sampling);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::llama_tokenize(ctx, client.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        llama_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { client.id + 1 }, false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.i_batch   = batch.n_tokens - 1;

                    LOG_TEE("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);

                    g_seq_id += 1;

                    // insert new requests one-by-one
                    //if (cont_batching) {
                    //    break;
                    //}
                }
            }
        }

        if (batch.n_tokens == 0) {
            break;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            //if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //    n_batch /= 2;
            //    i -= n_batch;
            //    continue;
            //}

            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = llama_sampling_sample(client.ctx_sampling, ctx, NULL, client.i_batch - i);

                llama_sampling_accept(client.ctx_sampling, ctx, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = llama_token_to_piece(ctx, id);

                client.response += token_str;
                client.sampled = id;

                //printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //        client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                // Brian edit: force model to stop on eos OR eot
                // Also make it so n_decoded  and not n_decoded + n_prompt is >= n_predict.
                //auto findStop = std::find(params.antiprompt.begin(), params.antiprompt.end(), client.response);

                bool foundStop = false;
                for (const auto& item : params.antiprompt) {
                    //printf("\nantiPrompt vec item = %s\n", item.c_str());
                    if (client.response.find(item) != std::string::npos) {
                    //if (client.response.find('\n') != std::string::npos) {
                        foundStop = true;
                        break;
                    }
                }
                
                if(client.response == " Yes" || client.response == " No"){
                    // Extract logits (trying to get probability)
                    auto   n_vocab = llama_n_vocab(model);
                    auto * logits_ptr  = llama_get_logits_ith(ctx, client.i_batch);

                    std::vector<float> logits;
                    logits.reserve(n_vocab);

                    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                        logits.emplace_back(logits_ptr[token_id]);
                    }

                    // Apply softmax to the extracted logits to get probabilities
                    std::vector<float> probabilities = softmax(logits);

                    // auto max_it = std::max_element(probabilities.begin(), probabilities.end());
                    // // Find the index of the maximum element
                    // int max_index = std::distance(probabilities.begin(), max_it);
                    // // Get the maximum value
                    // float max_prob = *max_it;
                    // // Get the string associated with it
                    // std::string max_str = llama_token_to_piece(ctx, max_index);

                    // Copy the client response
                    outFile4 << client.response << "\t";
                    float yesProb = -1.0;
                    float noProb = -1.0;

                    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                        std::string token_str = llama_token_to_piece(ctx, token_id);
                        if(token_str == " Yes") {
                            // Set yes probability
                            yesProb = probabilities[token_id]; 
                            //printf("Token ID: %d Token str: '%s' Logits: %.6f Probability: %.8f\n", token_id, token_str.c_str(), logits[token_id], probabilities[token_id]);
                        }else if (token_str == " No"){
                            // Set no probability
                            noProb = probabilities[token_id];
                            //printf("Token ID: %d Token str: '%s' Logits: %.6f Probability: %.8f\n", token_id, token_str.c_str(), logits[token_id], probabilities[token_id]);
                        }
                    }
                    // Write yes and no probs to file.
                    outFile4 << yesProb << "\t";
                    outFile4 << noProb << "\t";
                    outFile4 << quoteAndEscape(client.input, 20) << std::endl;
                }
                

                if (client.n_decoded > 2 &&
                        (llama_token_is_eog(model, id) || 
                        foundStop ||
                         (params.n_predict > 0 && client.n_decoded >= params.n_predict))) {
                    
                    // Brian edit: basic reverse prompt identifying the EOT or EOS tokens
                    const std::string eos_str = llama_token_to_piece(ctx, llama_token_eos(model));
                    const std::string eot_str = llama_token_to_piece(ctx, llama_token_eot(model));
                    printf("\nEOT string = '%s'\n", eot_str.c_str());
                    printf("\nEOS string = '%s'\n", eos_str.c_str());
                    printf("Client response (before chopping) = '%s'\n", client.response.c_str());
                    size_t pos;
                    if (eot_str.empty()) {
                        pos = client.response.find(eos_str);
                    } else{
                        const size_t pos_eos = client.response.find(eos_str);
                        const size_t pos_eot = client.response.find(eot_str);
                        pos = (pos_eos < pos_eot) ? pos_eos : pos_eot;
                    }
                    printf("\nEOS/EOT position = %zu\n", pos);

                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                    LOG_TEE("System:    %s\nInput:    \033[96m%s\n\033[0mResponse: \033[31m%s\033[0m\n\n",
                            ::trim(system).c_str(),
                            //::trim(prompts[promptNumber]).c_str(),
                            ::trim(client.input).c_str(),
                            ::trim(client.response).c_str());

                    n_total_prompt += client.n_prompt;
                    n_total_gen    += client.n_decoded;

                    client.seq_id = -1;
                }

                client.i_batch = -1;
            }
        }
    }

    // Close the file
    outFile4.close();

    const auto t_main_end = ggml_time_us();

    print_date_time();

    LOG_TEE("\n%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_TEE("External prompt file: \033[32m%s\033[0m\n", params.prompt_file.c_str());
    LOG_TEE("Model and path used:  \033[32m%s\033[0m\n\n", params.model.c_str());

    LOG_TEE("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Cache misses:        %6d\n", n_cache_miss);

    LOG_TEE("\n");

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
