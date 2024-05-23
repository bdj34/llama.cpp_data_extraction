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
#include <set>
#include <sstream>
#include <unordered_map>

std::string generatePreSystemPrompt(const std::string& promptFormat);
std::string generatePostSystemPrompt(const std::string& promptFormat);
std::string generatePreAnswer(const std::string& promptFormat, const std::string& answerType);\
//std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat);
std::string formatSystemPrompt(const std::string& systemPrompt, const std::string& promptFormat);
std::string quoteAndEscape(const std::string& input, bool quote);
//std::string createResponsePrompt(const std::string& systemPrompt, const std::string& modelThoughts, const std::string& promptFormat);

// Struct which will serve as the value of a key-value pair in an unordered map.
struct info {
    std::string inputText;
    std::string inputDate;
    std::string fullPrompt;
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

std::vector<std::string> k_prompts;

// static std::string default_system_thoughts =
// R"(The text provided is an excerpt from a medical note. You are responsible for building an accurate structured dataset from these notes. 
// In order to do so, determine the length of time (in months or years) between this note and the patient's original diagnosis with any of the following: 
// Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease. 
// If the duration from diagnosis to the encounter in the note is not obvious, consider the duration unknown 
// and a different medical note from the same patient can be used to determine diagnosis date. 
// It is important to be conservative and err on the side of unknown, waiting until the duration is clear and obvious before making a definitive call. 
// First, write out your reasoning in a single sentence. 
// Then, write your answer matching the following examples (examples: 'Answer: X months', 'Answer: Unknown', or 'Answer: X years').)";

// static std::string default_system_respond_preamble =
// R"(The text provided is an excerpt from a medical note. You are responsible for building an accurate structured dataset from these notes. 
// In order to do so, determine the length of time (in months or years) between this note and the patient's original diagnosis with any of the following: 
// Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease. 
// If the duration from diagnosis to the encounter in the note is not obvious, consider the duration 'Unknown' 
// and a different medical note from the same patient can be used to determine diagnosis date. 
// First, write out your reasoning in one sentence or less. 
// Then, write your answer matching the following examples (examples: 'Answer: X months', 'Answer: Unknown', or 'Answer: X years').)";

static std::string calYear_system = "The text provided is an excerpt from a medical note. You are responsible for building an accurate structured dataset from these notes. "
"In order to do so, determine the calendar year in which the patient was originally diagnosed with any of the following: "
"Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease. "
"If the calendar year of diagnosis cannot be confidently determined, consider the year 'Unknown' "
"and a different medical note from the same patient can be used to determine the diagnosis year. "
"This patient has several other notes containing information on their IBD history, so be conservative in answering. "
"Format your answer as in the following examples: 'Answer: XXXX' or 'Answer: Unknown'.";

static std::string duration_system = "The text provided is an excerpt from a medical note. You are responsible for building an accurate structured dataset from these notes. "
"In order to do so, determine the length of time (in years) between this note and the patient's original diagnosis with any of the following: "
"Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease. "
"If the duration from diagnosis to the encounter in the note is not obvious, consider the duration 'Unknown' "
"and a different medical note from the same patient can be used to determine diagnosis date. "
"Format your answer as in the following examples (examples: 'Answer: X years', 'Answer: Unknown').";

std::string generatePreSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return " [INST] ";
    } else if (promptFormat == "llama3") {
        return "<|start_header_id|>system<|end_header_id|>\n\n";
    } else if (promptFormat == "phi3") {
        return "<user>\n";
    //} else if (promptFormat == "phi3") {
    //    return "<system>\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePostSystemPrompt(const std::string& promptFormat) {
    if (promptFormat == "mistral") {
        return "\n<<<\n";
    } else if (promptFormat == "llama3") {
        return "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<<<\n";
    } else if (promptFormat == "phi3") {
        return "\n<<<\n";
    } else {
        throw std::runtime_error("Error: prompt format not recognized. Recognized options are: phi3, llama3, mistral.");
    }
}

std::string generatePreAnswer(const std::string& promptFormat, const std::string& answerType) {

    std::string question;
    if (answerType == "duration"){
        question = "What is the length of time (in years) between this note and the patient's original diagnosis with any of the following: "
       "Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease?";
    } else if (answerType == "calYear"){
        question = "To the nearest calendar year, when was the patient originally diagnosed with any of the following: "
       "Inflammatory Bowel Disease (IBD), colitis, proctitis, Ulcerative Colitis (UC) or Crohn's Disease? "
       "Be conservative, responding 'Unknown' if the exact year of original diagnosis cannot be easily determined from the note.";
    }

    if (promptFormat == "mistral") {
        return "\n>>>\n\n" + question + " [/INST] Answer:";
    } else if (promptFormat == "llama3") {
        return "\n>>>\n\n" + question + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nAnswer:";
    } else if (promptFormat == "phi3") {
        return "\n>>>\n\n" + question + "<|end|>\n <|assistant|> Answer:";
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

int main(int argc, char ** argv) {
    srand(1234);

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // Start with patient 0
    size_t patientNumber = 0;

    if(params.n_parallel != 1){
        throw std::runtime_error("Error: CPU mode only allows consecutive sequential processing (n_parallel = 1) for IBD hx at the moment.");
    }

    // What if we had more clients than n_parallel (# of IBD patients is # of clients, but only n_parallel are processed at a time)
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
    char dateTimeBuffer[30];  // Ensure the buffer is large enough for the format
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
    std::string outputFile = dirPath + "/output_" + params.answerType + "_" + dateTimeOutFile + ".txt";
    std::string consensusFile = dirPath + "/consensus_" + params.answerType + "_" + dateTimeOutFile + ".txt";

    std::vector<std::string> allPrompts;
    std::vector<std::string> allPatients;

    // load the prompts and patients from external files
    if (params.prompt.empty()) {
        throw std::runtime_error("Error: No prompts given");
    } else if (params.patients.empty()){
        throw std::runtime_error("Error: No patient IDs given");
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

        // Make sure we only run as many prompts as there are
        size_t n_prompts = allPrompts.size();
        if(n_seq + params.promptStartingNumber > n_prompts){
            n_seq -= params.promptStartingNumber;
        }

        // Print the prompts and write to outfile (only those equal to or after starting index)
        std::string tmpPrompt;
        for (const auto& prompt : allPrompts) {
            if(index >= params.promptStartingNumber){
                k_prompts.resize(index + 1);
                tmpPrompt = prompt + generatePreAnswer(params.promptFormat, params.answerType);
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

    // Check if the patients vector is sorted by ID (in some way, doesn't matter how it is sorted)
    // Also fill vector to tell us at what index each patient has their first note
    std::vector<size_t> firstPatientPrompts;
    firstPatientPrompts.push_back(0);
    for (size_t i = 1; i < allPatients.size(); ++i) {
        if (allPatients[i] == allPatients[i - 1]) {
            continue;
        }else{
            firstPatientPrompts.push_back(i);
        }
        // Check if the current string appears again later
        for (size_t j = i + 1; j < allPatients.size(); ++j) {
            if (allPatients[j] == allPatients[i - 1]) {
                throw std::runtime_error("Error: Patient ID file must be created so that repeated patient IDs are grouped together, not separated.");
            }
        }
    }

    size_t promptNumber = firstPatientPrompts[0];

    // Get the unique patients
    std::set<std::string> uniquePatients_set(allPatients.begin(), allPatients.end());
    std::vector<std::string> uniquePatients_vec(uniquePatients_set.begin(), uniquePatients_set.end());

    // Print the number of unique patients
    printf("Number of unique patients: %lu\n", uniquePatients_set.size());

    fprintf(stderr, "\n\n");
    fflush(stderr);

    const int n_ctx = llama_n_ctx(ctx);

    // Write format to the metadataFile
    std::string promptFormat_example = formatSystemPrompt("{System prompt here}", params.promptFormat) + "{Input text here}" + generatePreAnswer(params.promptFormat, params.answerType);
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
    outFile2 << "Output file format: {Y/N text} \\t {Yes Prob.} \\t {No Prob.} \\t {Full path report input (to make sure we have the right input mapped to the right output. \\n's and \\t's are escaped)}" << std::endl << std::endl;   
    outFile2 << "Model path: " << params.model << std::endl << std::endl;
    outFile2 << "Input file path: " << params.prompt_file << std::endl;
    outFile2 << "Reading from line " << params.promptStartingNumber << " to " << n_seq+params.promptStartingNumber-1 << " (zero-based index)" << std::endl << std::endl;
    outFile2 << quoteAndEscape(promptFormat_example, true) << std::endl << std::endl << "Prompt format tokenized:" << std::endl; // Adding newline for separation in file

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

    // Set system prompt (no formatting yet)
    std::string system;
    if(params.answerType == "calYear"){
        system = calYear_system;
    }else if(params.answerType == "duration"){
        system = duration_system;
    }
    if(!params.systemPrompt.empty()){
        system = params.systemPrompt;
    }

    // Write system prompt to the out file
    outFile2 << "System prompt: " << quoteAndEscape(system, true) << std::endl << std::endl; // Adding newline for separation in file


    // Format system prompt
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
    std::ofstream outFile3(outputFile.c_str());
    // Check if the file was opened successfully
    if (!outFile3) {
        std::cerr << "Failed to open the output out file." << std::endl;
        return 1; // Return with error code
    }

    // Define tmpResponses
    std::vector<std::string> tmpResponses;
    std::vector<std::string> consensusVec;

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

                    // // Extract logits (trying to get probability)
                    // auto * logits  = llama_get_logits_ith(ctx, pos-1);
                    // std::vector<llama_token_data> candidates;
                    // candidates.reserve(n_vocab);

                    // for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    //     candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
                    // }
                    

                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // Add client response to the tmpVec if not unknown
                    if(client.response != " Unknown"){
                        tmpResponses.push_back(client.response);
                    }
                    
                    // See if we have a consensus
                    std::unordered_map<std::string, size_t> entry_counts;

                    // Count the occurrences of each entry
                    for (const auto& entry : tmpResponses) {
                        ++entry_counts[entry];
                    }

                    // Find the entry with the maximum count
                    std::string consensus_entry;
                    size_t max_count = 0;
                    for (const auto& pair : entry_counts) {
                        if (pair.second > max_count) {
                            max_count = pair.second;
                            consensus_entry = pair.first;
                        }
                    }

                    size_t promptNumberOld = promptNumber;

                    // Check if the maximum count entry meets requirements
                    if ((static_cast<double>(max_count) / tmpResponses.size() >= params.minConsensusFraction) & (tmpResponses.size() >= params.n_minNotes)) {
                        std::printf("Consensus entry: %s\n", consensus_entry.c_str());
                        patientNumber += 1;
                        if(patientNumber < uniquePatients_vec.size()){
                            promptNumber = firstPatientPrompts[patientNumber];
                        }
                        tmpResponses.clear();
                        consensusVec.push_back(consensus_entry);

                    }else if(std::find(firstPatientPrompts.begin(), firstPatientPrompts.end(), promptNumber) != firstPatientPrompts.end()){
                        // If promptNumber is in firstPatient Prompts, wee've stumbled into a new patient
                        std::printf("No consensus entry for this patient\n");
                        patientNumber += 1;
                        tmpResponses.clear();
                        consensusVec.push_back("No consensus");
                    }


                    // Copy the client response and the input
                    outFile3 << client.response << "\t";

                    outFile3 << quoteAndEscape(client.input, false) << std::endl;

                    // Write N blank lines to the out file for the ones we skip
                    size_t N_blank_lines = promptNumber - promptNumberOld;
                    for (std::size_t i = 0; i < N_blank_lines; ++i) {
                        outFile3 << "Not run\t" << quoteAndEscape(k_prompts[promptNumberOld + i], false) << std::endl;
                        g_seq_id += 1;
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_TEE("System:    %s\nInput:    \033[96m%s\n\033[0mResponse: \033[31m%s\033[0m\n\n",
                            ::trim(system).c_str(),
                            //::trim(prompts[promptNumber]).c_str(),
                            ::trim(client.input).c_str(),
                            ::trim(client.response).c_str());

                    // LOG_TEE("\033[31mClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d \033[0m \nInput:    %s\n\033[35mResponse: %s\033[0m\n\n",
                    //         client.id, client.seq_id, n_seq, client.n_prompt, client.n_decoded,
                    //         (t_main_end - client.t_start_prompt) / 1e6,
                    //         (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                    //         n_cache_miss,
                    //         ::trim(client.input).c_str(),
                    //         ::trim(client.response).c_str());

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

    // Open consensus file to write to it
    std::ofstream outFile4(consensusFile.c_str());
    // Check if the file was opened successfully
    if (!outFile4) {
        std::cerr << "Failed to open the output out file." << std::endl;
        return 1; // Return with error code
    }

    // Write each string from the vector to the file
    for (const auto& str : consensusVec) {
        outFile4 << str << "\n";
    }
    outFile4.close();

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
                }

                // Create a set for acceptable answers
                std::set<std::string> acceptableAnswers;
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
