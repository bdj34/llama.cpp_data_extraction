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
std::string generatePostSystemPrompt(const std::string& promptFormat, const std::string& extractionDx);
std::string generatePreAnswer(const std::string& promptFormat, const std::string& extractionDx);
std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat, const std::string& extractionDx);
std::string escapeNewLines(const std::string& input);
std::string convertEscapedNewlines(const std::string& input);

// trim whitespace from the beginning and end of a string. Only used for printing 
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

std::string convertEscapedNewlines(const std::string& input) {
    std::string output = input;
    size_t pos = 0;
    
    // Find and replace all occurrences of "\\n" with "\n"
    while ((pos = output.find("\\n", pos)) != std::string::npos) {
        output.replace(pos, 2, "\n");
        pos += 1; // Move past the newly inserted '\n' to avoid infinite loop
    }
    
    return output;
}

std::vector<std::string> k_prompts;

std::string crohns_system = "You are provided with excerpts of medical notes for a patient who may have Crohn's colitis."
" Your task is to determine whether they have Crohn's disease which affects the colon."
" You are not a clinician. Do not make diagnostic judgments. Only extract information on diagnoses reported in the notes."
" First, provide a one sentence summary from the notes about the diagnosis."//and/or direct quotes from the notes about the diagnosis."
" Then, answer Yes if there is evidence of Crohn's disease causing colitis or affecting the colon (Crohn's colitis), or No if there is no evidence of Crohn's colitis."
" If the patient has Crohn's disease but there is no evidence that it affects their colon, answer No - Crohn's without colitis."
" If the diagnosis is undecided between Crohn's Disease and Ulcerative Colitis (UC), answer Undecided between UC and Crohn's."
" If the diagnosis is Ulcerative Colitis or Ulcerative Proctitis, answer No - UC."
" If the diagnosis is neither UC nor Crohn's, answer No."
" If the information is insufficient, answer Insufficient information or Unknown."
" Provide your confidence in the answer (low, medium, high, certain)."
" Also, indicate whether the diagnosis has been confirmed by colonoscopy, endoscopy or pathology."
" Finally, indicate if the exact date of diagnosis is stated in the notes."
" Format your answer as follows:\n"
"Summary from notes: {One sentence summary from the notes}\n"
"Answer: {Your answer}. Confidence: {Low, Medium, High, or Certain}\n"
"Pathology or endoscopy confirmed: {Yes or No}\n"
"Exact diagnosis date stated: {Yes or No}";

std::string crc_system = 
"The text provided is a pathology report, with samples originating from the colon or rectum unless specified otherwise."
" We are interested in identifying whether invasive adenocarcinoma (stage greater than or equal to 1) is present in *any* colon or rectal sample."
" Without definite invasion identified, conditions such as 'high-grade dysplasia', 'in-situ [adeno]carcinoma', or 'intramucosal [adeno]carcinoma' are not typically classified as invasive adenocarcinoma."
" If the sample is classified as having adenocarcinoma without further specification, this typically implies invasive adenocarcinoma."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?";

std::string advNeo_system = 
"The text provided is a pathology report, with samples originating from the colon or rectum unless specified otherwise."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that the patient has"
" adenocarcinoma (including in-situ or intramucosal adenocarcinoma) or high-grade dysplasia"
" in any colon or rectal sample?";

std::string generatePreSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return "[INST] "; // Not sure whether to include preceding space...
    } else if (promptFormat == "llama3") {
        return "<|start_header_id|>system<|end_header_id|>\n\n";
    } else if (promptFormat == "phi3") {
        return "<|user|>\n";
    } else if (promptFormat == "gemma2") {
        return "<start_of_turn>user\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: gemma2, phi3, llama3, mistral.");
    }
}

std::string generatePostSystemPrompt(const std::string& promptFormat, const std::string& extractionDx) {

    std::string postSys;
    if (extractionDx == "crohns"){
        postSys = "Excerpts:\n";
    } else if (extractionDx == "crc" || extractionDx == "advNeo"){
        postSys = "<<<\nPathology report:\n";
    } else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, crohns, advNeo. lgd coming soon.");
    }


    if (promptFormat == "mistral") {
        return "\n\n" + postSys;
    } else if (promptFormat == "llama3") {
        return "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n" + postSys;
    } else if (promptFormat == "phi3") {
        return "\n\n" + postSys;
    } else if (promptFormat == "gemma2") {
        return "\n\n" + postSys;
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: gemma2, phi3, llama3, mistral.");
    }
}

std::string crohns_question = "Question: Does the patient have Crohn's colitis?";
std::string crc_question = ">>>\n\nDoes the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?";
std::string advNeo_question = ">>>\n\nDoes the pathology report indicate that the patient has"
" adenocarcinoma (including in-situ or intramucosal adenocarcinoma) or high-grade dysplasia"
" in the colon or rectum?";

std::string crohns_preAnswer = "Summary from notes:";
std::string crc_preAnswer = "Answer:";
std::string advNeo_preAnswer = "Answer:";

std::string generatePreAnswer(const std::string& promptFormat, const std::string& extractionDx) {

    std::string question;
    std::string preAnswer;

    if (extractionDx == "crohns"){
        question = crohns_question;
        preAnswer = crohns_preAnswer;
    } else if (extractionDx == "crc"){
        question = crc_question;
        preAnswer = crc_preAnswer;
    } else if(extractionDx == "advNeo"){
        question = advNeo_question;
        preAnswer = advNeo_preAnswer;
    }else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, crohns, advNeo. lgd coming soon.");
    }

    if (promptFormat == "mistral") {
        return "\n" + question + " [/INST] " + preAnswer;
    } else if (promptFormat == "llama3") {
        return "\n" + question + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n" + preAnswer;
    } else if (promptFormat == "phi3") {
        return "\n" + question + "<|end|>\n<|assistant|>\n" + preAnswer;
    } else if (promptFormat == "gemma2") {
        return "\n" + question + "<end_of_turn>\n<start_of_turn>model\n" + preAnswer;
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: gemma2, phi3, llama3, mistral.");
    }
}

std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat, const std::string& extractionDx) {
    std::string prePrompt = generatePreSystemPrompt(promptFormat);
    std::string postSystem = generatePostSystemPrompt(promptFormat, extractionDx);

    return prePrompt + systemPrompt + postSystem;
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

    std::string ptID;

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
    
    std::string system;
    if (params.extractionType == "crohns"){
        system = crohns_system;
    } else if (params.extractionType == "crc"){
        system = crc_system;
    } else if(params.extractionType == "advNeo"){
        system = advNeo_system;
    }else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, crohns, advNeo. lgd coming soon.");
    }

    // Get the prompt Number we start at
    size_t promptNumber = params.promptStartingNumber;

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

    // initialize the model 

    llama_model_params model_params = llama_model_params_from_gpt_params(params);

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    //llama_model * model = NULL;
    //llama_context * ctx = NULL;

    // initialize the context

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);

    // Set file names
    std::string dirPath = params.outDir;
    std::string inputFile = dirPath + "/inputTextNoFormatting_" + dateTimeOutFile + ".txt";
    std::string metadataFile = dirPath + "/metadata_" + dateTimeOutFile + ".txt";
    std::string outputFile = dirPath + "/output_" + dateTimeOutFile + ".txt";

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
                tmpPrompt = prompt + generatePreAnswer(params.promptFormat, params.extractionType);
                k_prompts[index] = tmpPrompt;

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

    // Write format to the metadataFile
    std::string promptFormat_example = formatSystemPrompt(system, params.promptFormat, params.extractionType) + convertEscapedNewlines(k_prompts[params.promptStartingNumber]);
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
    outFile2 << "Start date-time: " << dateTimeOutFile << std::endl;
    outFile2 << "Output file format (tab-separated): {Model answer, with newlines escaped}\t{Patient ID or SurgPathID}" << std::endl << std::endl;   
    outFile2 << "Model path: " << params.model << std::endl << std::endl;
    outFile2 << "Input file path: " << params.prompt_file << std::endl;
    outFile2 << "Patient ID file path (if applicable): " << params.patient_file << std::endl;
    outFile2 << "Reading from line " << params.promptStartingNumber << " to " << n_seq+params.promptStartingNumber << " (zero-based index)" << std::endl << std::endl;
    outFile2 << "Prompt format: " << params.promptFormat << std::endl;
    outFile2 << "Prompt format example (no escaping):" << std::endl; 
    outFile2 << promptFormat_example << std::endl << std::endl << std::endl << "Prompt format tokenized (including BOS token):" << std::endl; // Adding newline for separation in file

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

    // Initialize system prompt token vec
    std::vector<llama_token> tokens_system;
    // Format system prompt
    std::string k_system = formatSystemPrompt(system, params.promptFormat, params.extractionType);
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

            //LOG_TEE("%s: clearing the KV cache\n", __func__);
        }

        // insert new sequences for decoding
        if (cont_batching || batch.n_tokens == 0) {
            for (auto & client : clients) {
                if (client.seq_id == -1 && g_seq_id < n_seq) {
                    client.seq_id = g_seq_id;

                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    client.input    = convertEscapedNewlines(k_prompts[promptNumber]);
                    if(!params.patient_file.empty()){
                        client.ptID = allPatients[promptNumber];
                    }

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
                        LOG_TEE("\n\n\033[0mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);
                    }else{
                        LOG_TEE("\n\n\033[0mClient %3d, Patient %s, seq %4d, started decoding ...\033[0m\n", client.id, client.ptID.c_str(), client.seq_id);
                    }
                    
                    g_seq_id += 1;
                }
            }
        }

        if (batch.n_tokens == 0) {
            break;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {

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

                bool foundStop = false;
                for (const auto& item : params.antiprompt) {
                    if (client.response.find(item) != std::string::npos) {
                        foundStop = true;
                        break;
                    }
                }

                // Determine when to stop generating
                if (client.n_decoded > 0 &&
                        (llama_token_is_eog(model, id) || 
                        foundStop ||
                         (params.n_predict > 0 && client.n_decoded >= params.n_predict))) {
                    
                    // Brian edit: basic reverse prompt identifying the EOT or EOS tokens
                    const std::string eos_str = llama_token_to_piece(ctx, llama_token_eos(model));
                    int32_t eot_token = llama_token_eot(model);

                    size_t pos;
                    if (eot_token == -1) {
                        pos = client.response.rfind(eos_str);
                    } else{
                        const std::string eot_str = llama_token_to_piece(ctx, llama_token_eot(model));
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
                    }

                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // Copy the client response and the ptID to the output file
                    outFile3 << escapeNewLines(client.response);
                    if(!client.ptID.empty()){
                        outFile3 << "\t" << client.ptID;
                    }
                    outFile3 << std::endl;

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_TEE("\033[0m \nInput:\n\033[96m%s\n\033[91m%s\033[0m\n\033[92mJust completed: Patient: %s, sequence %3d of %3d, prompt: %4d tokens, response: %4d tokens, time: %5.2f seconds, speed %5.2f t/s",
                            //escapeNewLines(client.input).c_str(),
                            client.input.c_str(),
                            client.response.c_str(),
                            client.ptID.c_str(), client.seq_id, n_seq, client.n_prompt, client.n_decoded,
                            (t_main_end - client.t_start_prompt) / 1e6,
                            (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6);
                            // n_cache_miss,
                            //k_system.c_str(),
                            //::trim(prompts[promptNumber]).c_str());
                    
                    LOG_TEE("\nJust completed Patient: %s",
                        client.ptID.c_str());

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

    // Reopen the metadata file in append mode
    std::ofstream metaFile(metadataFile, std::ios::app);  // Append mode

    // Check if the file was opened successfully
    if (!metaFile) {
        std::cerr << "Failed to open the metadata out file." << std::endl;
        return 1; // Return with error code
    }

    const auto t_main_end = ggml_time_us();

    metaFile << "Total runtime (seconds): " << (t_main_end - t_main_start)/(1e6) << std::endl;

    metaFile.close();

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
