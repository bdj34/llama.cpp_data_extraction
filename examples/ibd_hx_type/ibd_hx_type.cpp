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
#include <fstream>
#include <unordered_set>
#include <regex>

std::string generatePreSystemPrompt(const std::string& promptFormat);
std::string generatePostSystemPrompt(const std::string& promptFormat);
std::string generatePreAnswer(const std::string& promptFormat);
std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat);
std::string quoteAndEscape(const std::string& input, bool quote);
std::string escapeNewLines(const std::string& input);
std::string getLastLine(const std::string& str);
std::string getFirstLine(const std::string& str);

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

std::vector<std::string> k_prompts;

std::string calYear_system = "You are an expert medical chart reviewer creating a structured dataset from medical notes. "
"The excerpts below are from one individual patient's medical record. "
"Examine the provided medical notes to extract information regarding the year of colitis diagnosis "
"and the specific type of colitis, if colitis has been diagnosed. If there is no mention of colitis, clearly state 'No colitis' "
"in your response. Use the information directly stated in the notes without making any diagnostic judgments. Identify "
"the type of colitis diagnosed, such as Ulcerative Colitis (UC), Ulcerative Proctitis, Ulcerative Pancolitis, "
"Crohn's Disease, Ischemic Colitis, Infectious Colitis, C. difficile Colitis, "
"Microscopic Colitis, Drug-induced Colitis, etc., or specify if the "
"notes do not mention colitis. "
// "Crohn's Disease may affect the colon, but can be found anywhere from the mouth to the anus. "
// "If the notes diagnose Crohn's Disease that *does not affect the colon*, respond 'Crohn's Disease without colonic involvement'. "
// "If the notes diagnose Crohn's Disease that *does affect the colon*, respond 'Crohn's Disease with colonic involvement'. "
// "If it is *unclear* whether the Crohn's Disease affects the colon, respond 'Crohn's Disease with unknown colonic involvement'. "
"If the diagnosis is undecided between Crohn's Disease and Ulcerative Colitis, respond 'IBD colitis'. "
"If colitis is identified, but the type is unknown, respond 'Unspecified Colitis'. "
"Determine the original year the diagnosis was made, if available. If the year of original diagnosis is not clear, respond 'Unknown'. "
"Your responses should be based solely on the information provided, without assuming details not explicitly stated. "
"Also provide your confidence (Low, Medium, High, Certain) in the year and type of diagnosis. "
"Now, read the excerpts and follow the instructions below, keeping the previous points in mind.";


std::string generatePreSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return " [INST] ";
    } else if (promptFormat == "llama3") {
        return "<|start_header_id|>system<|end_header_id|>\n\n";
    } else if (promptFormat == "phi3") {
        return "<|user|>\n";
    } else if (promptFormat == "gemma2") {
        return "<start_of_turn>user\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePostSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return "\n";
    } else if (promptFormat == "llama3") {
        return "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n";
    } else if (promptFormat == "phi3") {
        return "\n";
    } else if (promptFormat == "gemma2") {
        return "\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePreAnswer(const std::string& promptFormat) {

    std::string question = "### Instructions\n"
    "Year of Original Diagnosis: Determine the original year of diagnosis. "
    "We are only interested in the *original* diagnosis year. Be conservative and respond 'Unknown' when the exact year of the original diagnosis is unclear.\n"
    "Diagnosis Type: Note the diagnosis as per the notes.\n"
    "Format your answer as follows:\n"
    "Reasoning and Evidence from Notes about *Original* Year of Diagnosis: {Direct quotes or summaries from the notes and your reasoning}\n"
    "Year of *Original Diagnosis* (YYYY): {'Unknown' or Year}, Confidence in Year: {Confidence}\n"
    "Reasoning and Evidence from Notes about Diagnosis Type: {Direct quotes or summaries from the notes and your reasoning}\n"
    "Diagnosis Type: {Type}, Confidence in Type: {Confidence}";

    if (promptFormat == "mistral") {
        return "\n\n" + question + " [/INST] Reasoning and Evidence from Notes about *Original* Year of Diagnosis:";
    } else if (promptFormat == "llama3") {
        return "\n" + question + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReasoning and Evidence from Notes about *Original* Year of Diagnosis:";
    } else if (promptFormat == "phi3") {
        return "\n" + question + "<|end|>\n<|assistant|>\nReasoning and Evidence from Notes about *Original* Year of Diagnosis:";
    } else if (promptFormat == "gemma2") {
        return "\n" + question + "<end_of_turn>\n<start_of_turn>model\nReasoning and Evidence from Notes about *Original* Year of Diagnosis:";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat) {
    std::string prePrompt = generatePreSystemPrompt(promptFormat);
    std::string postSystem = generatePostSystemPrompt(promptFormat);
    
    return prePrompt + systemPrompt + postSystem;
}

// Function to escape quotes, newlines, and tabs, and optionally enclosethe string in quotes
std::string quoteAndEscape(const std::string& input, bool quote) {

    std::string output;
    if(quote){
        output = "\"";  // Start with an opening quote
    }
    for (char ch : input) {
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
    if(quote){
        output += "\"";  // End with a closing quote
    }
    return output;
}

std::string escapeNewLines(const std::string& input) {

    std::string output;

    for (char ch : input) {
        switch (ch) {
            case '\n':
                output += "\\n";   // Escape newlines
                break;
            default:
                output += ch;
        }
    }
    return output;
}

std::string getLastLine(const std::string& str) {
    std::regex lineRegex("(.*?)(\n|$)");
    std::smatch match;
    std::string lastLine;
    
    // Initialize an iterator to search through all regex matches
    auto begin = std::sregex_iterator(str.begin(), str.end(), lineRegex);
    auto end = std::sregex_iterator();

    // Iterate through all matches to find the last non-empty line
    for (auto it = begin; it != end; ++it) {
        if (!it->str(1).empty()) {  // Ensure the captured line is not empty
            lastLine = it->str(1);
        }
    }

    return lastLine;
}

std::string getFirstLine(const std::string& str) {
    std::regex lineRegex("^.*?(?:\n|$)");
    std::smatch match;
    
    if (std::regex_search(str, match, lineRegex)) {
        return match[0].str();
    }

    return ""; // Return empty string if no match found or input string is empty
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

    std::string ICN;

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

int main(int argc, char ** argv) {
    srand(1234);

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // Get the prompt Number we start at
    size_t promptNumber = params.promptStartingNumber;

    if(params.testing_mode){
        if(promptNumber != 0){
            throw std::runtime_error("Error: Testing mode is not compatible with nonzero prompt number. Answer key and prompts will not match.");
        }
        if(params.n_parallel != 1){
            throw std::runtime_error("Error: Testing mode is not compatible with multiple simultaneous requests. Answer key and input prompts may not match as client requests may change order.");
        }
    }

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    // dedicate one sequence to the system prompt
    params.n_parallel += 1;

    // requests to simulate
    int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;

    const bool dump_kv_cache = params.dump_kv_cache;

    // Get current time
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);
    // Buffer to hold the date-time format
    char dateTimeBuffer[60];  // Ensure the buffer is large enough for the format
    // Format the date and time with strftime
    strftime(dateTimeBuffer, sizeof(dateTimeBuffer), "%Y-%m-%d_%H-%M-%S", now_tm);
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
    std::string metadataFile = dirPath + "/metadata_" + dateTimeOutFile + ".txt";
    std::string outputFile = dirPath + "/output_concat_" + dateTimeOutFile + ".txt";

    std::vector<std::string> allPrompts;
    std::vector<std::string> allPatients;
    // load the prompts from an external file if there are any
    if (params.prompt.empty()) {
        throw std::runtime_error("Error: No prompts given");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        size_t index = 0;
        printf("\n\033[32mNow printing the external prompt file starting with line %zu from %s\033[0m\n\n", params.promptStartingNumber, params.prompt_file.c_str());

        // Create and open a text file
        std::ofstream outFile1(inputFile.c_str());

        // Check if the file was opened successfully
        if (!outFile1) {
            std::cerr << "Failed to open the input prompt out file." << std::endl;
            return 1; // Return with error code
        }

        allPrompts = split_string(params.prompt, '\n');
        allPatients = split_string(params.patients, '\n');



        // Print the prompts and write to outfile (only those equal to or after starting index)
        std::string tmpPrompt;
        for (const auto& prompt : allPrompts) {
            if(index >= params.promptStartingNumber){
                k_prompts.resize(index + 1);
                tmpPrompt = prompt + generatePreAnswer(params.promptFormat);
                k_prompts[index] = tmpPrompt;

                printf("%zu prompt: %s\n", index, tmpPrompt.c_str());

                // Write each prompt to the out file
                outFile1 << prompt << std::endl; // Adding newline for separation in file
            }
            index++;
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
    outFile2 << "Output file format: {Year}, {Confidence}, {Resolution}, {Patient ID (where applicable)}" << std::endl << std::endl;   
    outFile2 << "Model path: " << params.model << std::endl << std::endl;
    outFile2 << "Input file path: " << params.prompt_file << std::endl;
    outFile2 << "Patient ID file path (if applicable): " << params.patient_file << std::endl;
    outFile2 << "Reading from line " << params.promptStartingNumber << " to " << n_seq+params.promptStartingNumber << " (zero-based index)" << std::endl << std::endl;
    outFile2 << "Prompt format example (no escaping):" << std::endl; 
    outFile2 << promptFormat_example << std::endl << std::endl << "Prompt format tokenized:" << std::endl; // Adding newline for separation in file

    // Iterate through the vector and write each element to the file
    for (size_t i = 0; i < tokens_format.size(); ++i) {
        outFile2 << tokens_format[i] << "\t";
    }
    outFile2 << std::endl << std::endl;

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        client.ctx_sampling = llama_sampling_init(params.sparams);
    }

    std::vector<llama_token> tokens_system;

    // Write system prompt to the out file
    outFile2 << "System prompt: " << quoteAndEscape(calYear_system, true) << std::endl << std::endl; // Adding newline for separation in file

    // Format system prompt
    std::string k_system = formatSystemPrompt(calYear_system, params.promptFormat);

    // Print the string
    printf("System prompt: %s\n", k_system.c_str());
    tokens_system = ::llama_tokenize(ctx, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();

    llama_seq_id g_seq_id = params.promptStartingNumber;

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);

    const auto t_main_start = ggml_time_us();

    LOG_TEE("%s: Simulating parallel requests from %d patients:\n", __func__, n_seq);
    LOG_TEE("%s: n_parallel (number of simultaneous requests) = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, cont_batching, n_tokens_system);
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
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }

        LOG_TEE("\n");
    }

    LOG_TEE("Processing requests ...\n\n");

    // Open output file to write to
    std::ofstream outFile3(outputFile.c_str());
    // Check if the file was opened successfully
    if (!outFile3) {
        std::cerr << "Failed to open the output out file." << std::endl;
        return 1; // Return with error code
    }

    while (true) {
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            llama_kv_cache_dump_view_seqs(kvc_view, 40);
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
                    if(!params.patient_file.empty()){
                        client.ICN = allPatients[promptNumber];
                    }
                    //printf("%zu\n", promptNumber);
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

                    if(params.patient_file.empty()){
                        LOG_TEE("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);
                    }else{
                        LOG_TEE("\033[31mClient %3d, Patient %s, seq %4d, started decoding ...\033[0m\n", client.id, client.ICN.c_str(), client.seq_id);
                    }
                    
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
            //printf("%d\n", i);

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
                //printf("%s", token_str.c_str());

                client.sampled = id;

                // Brian edit: force model to stop on eos OR eot
                // Also make it so n_decoded  and not n_decoded + n_prompt is >= n_predict.
                //auto findStop = std::find(params.antiprompt.begin(), params.antiprompt.end(), client.response);

                bool foundStop = false;
                for (const auto& item : params.antiprompt) {
                    if (client.response.find(item) != std::string::npos) {
                        foundStop = true;
                        break;
                    }
                }

                if (client.n_decoded > 2 &&
                        (llama_token_is_eog(model, id) || 
                        foundStop ||
                         (params.n_predict > 0 && client.n_decoded >= params.n_predict))) {
                    
                    // Brian edit: basic reverse prompt identifying the EOT or EOS tokens
                    const std::string eos_str = llama_token_to_piece(ctx, llama_token_eos(model));
                    printf("\nEOS string = '%s'\n", eos_str.c_str());

                    int32_t eot_token = llama_token_eot(model);
                    printf("EOT token: %d\n", llama_token_eot(model));

                    printf("Client response (before chopping) = '%s'\n", client.response.c_str());

                    //std::string lastLine = getLastLine(client.response);
                    //std::string firstLine = getFirstLine(client.response);

                    size_t pos;
                    if (eot_token == -1) {
                        pos = client.response.rfind(eos_str);
                    } else{
                        const std::string eot_str = llama_token_to_piece(ctx, llama_token_eot(model));
                        printf("\nEOT string = '%s'\n", eot_str.c_str());
                        const size_t pos_eos = client.response.rfind(eos_str);
                        const size_t pos_eot = client.response.rfind(eot_str);
                        if (pos_eos == std::string::npos && pos_eot == std::string::npos) {
                            pos = std::string::npos;
                        } else if (pos_eos == std::string::npos) {
                            pos = pos_eot;
                        } else if (pos_eot == std::string::npos) {
                            pos = pos_eos;
                        } else {
                            pos = (pos_eos > pos_eot) ? pos_eos : pos_eot;
                        }
                        //pos = (pos_eos > pos_eot) ? pos_eos : pos_eot;
                        //pos = pos_eos;
                    }
                    printf("\nEOS/EOT position = %zu\n", pos);

                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    //printf("\nlastLine = '%s'\n", lastLine.c_str());

                    // Copy the client response and the input
                    outFile3 << escapeNewLines(client.response) << "\t";
                    if(!client.ICN.empty()){
                        outFile3 << client.ICN << std::endl;
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_TEE("System:    %s\nInput:    \033[96m%s\n\033[0mResponse: \033[31m%s\033[0m\n\n",
                            ::trim(calYear_system).c_str(),
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
    outFile3.close();

    // If in testing mode, compare answers to answer key
    int matchCount = 0;
    if(params.testing_mode){

        std::ifstream trueAnswerFile(params.answerKey);
        std::ifstream modelAnswerFile(outputFile);
        
        
        std::string trueAnswerLine, modelAnswerLine;

        int lineNumber = 0;

        // Check corresponding lines from both files
        if (trueAnswerFile.is_open() && modelAnswerFile.is_open()) {
            while (getline(trueAnswerFile, trueAnswerLine) && getline(modelAnswerFile, modelAnswerLine)) {
                lineNumber++;
                std::istringstream iss(trueAnswerLine);
                std::string acceptableAnswer;
                bool matchFound = false;

                // Process the model answer line
                std::istringstream modelISS(modelAnswerLine);
                std::string firstModelAnswer;
                if (getline(modelISS, firstModelAnswer, '\t')) {  // Extract the first tab-separated entry
                    // Remove leading space if present
                    if (!firstModelAnswer.empty() && firstModelAnswer[0] == ' ') {
                        firstModelAnswer.erase(0, 1);
                    }

                    // printf("%s\n", firstModelAnswer.c_str());

                    // Use regex to find the first set of digits
                    // std::regex pattern(R"((\d+)|Unknown)");
                    // std::smatch match;
                    // if (std::regex_search(firstModelAnswer, match, pattern)) {
                    //     firstModelAnswer = match.str(0);
                    // } else {
                    //     firstModelAnswer = "Invalid"; // Handle the case where no digits are found
                    // }

                    // Check if "Unknown" is present
                    std::regex pattern(R"(Unknown)");
                    std::smatch match;
                    if (std::regex_search(firstModelAnswer, match, pattern)) {
                        // "Unknown" is present, use it
                        firstModelAnswer = "Unknown";
                    } else {
                        // Use regex to find the first set of digits
                        std::regex pattern(R"(\d+)");
                        std::smatch match;
                        if (std::regex_search(firstModelAnswer, match, pattern)) {
                            firstModelAnswer = match.str(0);
                        } else {
                            firstModelAnswer = "Invalid"; // Handle the case where no digits are found
                        }
                    }

                    
                    //printf("%s\n", firstModelAnswer.c_str());
                }

                // Create a set for acceptable answers
                std::unordered_set<std::string> acceptableAnswers;
                while (getline(iss, acceptableAnswer, '\t')) {
                    acceptableAnswers.insert(acceptableAnswer);
                }

                // Check if the model answer matches any acceptable answers
                if (acceptableAnswers.find(firstModelAnswer) != acceptableAnswers.end()) {
                    matchFound = true;
                }

                // Output the result for this line
                if (matchFound) {
                    matchCount++;
                    //std::cout << "Line " << lineNumber << ": Match found." << std::endl;
                } else {
                    //std::cout << "Line " << lineNumber << ": No match found." << std::endl;
                }
            }
            trueAnswerFile.close();
            modelAnswerFile.close();

            //std::cout << "Total matching lines: " << matchCount << std::endl;
        } else {
            std::cerr << "Unable to open one or both files" << std::endl;
            return 1;
        }
        
    }

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

    if(params.testing_mode){
        LOG_TEE("Accuracy:        %6d / %6d\n", matchCount, n_seq);
    }

    // Reopen the metadata file in append mode
    std::ofstream metaFile(metadataFile, std::ios::app);  // Append mode

    if (metaFile.is_open()) {
        if(params.testing_mode){
            metaFile << "Accuracy: " <<  matchCount << " / " << n_seq << std::endl;
        }
        metaFile << "Runtime: " << (t_main_end - t_main_start) / 1e6 << " seconds" << std::endl;
        metaFile.close();
    } else {
        std::cerr << "Unable to open metadata file for appending." << std::endl;
    }

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
