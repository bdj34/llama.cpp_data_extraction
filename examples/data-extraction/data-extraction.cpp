// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
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
std::string addFieldToJsonArray(const std::string& jsonArray, const std::string& newFieldName, const std::string& newFieldValue);

// Function to add a field to each entry of the JSON array
std::string addFieldToJsonArray(const std::string& jsonArray, const std::string& newFieldName, const std::string& newFieldValue) {
    std::string result;
    size_t pos = 0;

    while (pos != std::string::npos) {
        // Find the end of the next JSON object
        size_t endPos = jsonArray.find('}', pos);

        if (endPos == std::string::npos) {
            break; // No more entries
        }

        // Extract the JSON object
        std::string jsonObject = jsonArray.substr(pos, endPos - pos + 1);

        // Add new field before the closing '}'
        if (jsonObject.back() == '}') {
            // Remove the last closing brace '}'
            jsonObject.pop_back();

            // Add comma if this is not an empty object
            if (jsonObject.length() > 1) {
                jsonObject += ",";
            }

            // Add the new field
            jsonObject += "\"" + newFieldName + "\":\"" + newFieldValue + "\"}";

            // Append the modified object to the result
            result += jsonObject;
        }

        // Move the position past this object
        pos = endPos + 1;
    }

    // Wrap the result in array brackets
    return result;
}

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

// std::string ibd_system = "You are provided with excerpts of medical notes for a patient who may have colitis caused by Inflammatory Bowel Disease (IBD)."
// " Your task is to determine whether they have IBD which affects the colon (IBD colitis)."
// " You are not a clinician. Do not make diagnostic judgments. Only extract information on diagnoses reported in the notes."
// " First, provide a one sentence summary from the notes about the diagnosis."//and/or direct quotes from the notes about the diagnosis."
// " Then, provide your IBD colitis answer based on the following instructions:\n"
// "   - If the patient has Crohn's disease that affects their colon, answer Crohn's colitis.\n"
// "   - If the patient has Crohn's disease but there is no evidence that it affects their colon, answer Crohn's without colitis.\n"
// "   - If the diagnosis is undecided between Crohn's Disease and Ulcerative Colitis (UC), answer Undecided between UC and Crohn's.\n"
// "   - If the diagnosis is Ulcerative Colitis, answer Ulcerative Colitis.\n"
// "   - If the diagnosis is Ulcerative Proctitis (UC confined to rectum), answer Ulcerative proctitis.\n"
// "   - If the diagnosis is another type of colitis, answer {colitis type}.\n"
// "   - If the information is insufficient, answer Insufficient information or Unknown.\n"
// "Then provide your confidence in the answer (low, medium, high, certain)."
// " Also, indicate whether the diagnosis has been confirmed by colonoscopy, endoscopy or pathology."
// " Finally, indicate if the exact date of diagnosis is stated in the notes."
// " Format your answer as follows:\n"
// "Summary from notes: {One sentence summary from the notes}\n"
// "IBD colitis answer: {Your answer}. Confidence: {Low, Medium, High, or Certain}\n"
// "Pathology or endoscopy confirmed: {Yes or No}\n"
// "Exact date of original colitis diagnosis stated: {Yes or No}";

std::string ibd_system = 
"You are provided with excerpts of medical notes for a patient who may have colitis caused by Inflammatory Bowel Disease (IBD)."
" Your task is to identify whether the patient has IBD affecting the colon (IBD colitis), based solely on information reported in the notes."
" You are not a clinician and must not make diagnostic judgments. Only summarize and classify based on the evidence provided in the notes."
"\n\n"
"### Instructions:\n"
"1. Extract and summarize relevant information from the notes related to IBD diagnoses (two sentences or less).\n"
"2. Use the following guidelines to classify the diagnosis:\n"
"   - If the patient has Crohn's disease affecting the colon, answer 'Crohn's colitis'.\n"
"   - If the patient has Crohn's disease but no evidence of colon involvement, answer 'Crohn's without colitis'.\n"
"   - If the diagnosis is undecided between Crohn's Disease and Ulcerative Colitis (UC), answer 'Undecided between UC and Crohn's'.\n"
"   - If the diagnosis is Ulcerative Colitis, answer 'Ulcerative Colitis'.\n"
"   - If the diagnosis is Ulcerative Proctitis (UC confined to the rectum), answer 'Ulcerative proctitis'.\n"
"   - If the diagnosis is another type of colitis, specify as '{colitis type}'.\n"
"   - If the information is insufficient to make a determination, answer 'Insufficient information' or 'Unknown'.\n"
"\n"
"3. Indicate your confidence level in the answer (Low, Medium, High, or Certain).\n"
"4. Indicate whether the diagnosis has been confirmed by colonoscopy, endoscopy, or pathology.\n"
"5. State if the exact date of the original colitis diagnosis is provided in the notes.\n"
"\n"
"### Format your response as follows:\n"
"Summary from notes: {Provide a concise one or two sentence summary of the relevant diagnosis information from the notes.}\n"
"IBD colitis answer: {Your answer}. Confidence: {Low, Medium, High, or Certain}\n"
"Pathology or endoscopy confirmed: {Yes or No}\n"
"Exact date of original colitis diagnosis stated: {Yes or No}\n"
"\n"
"### Important points:\n"
"- Do not assume information that is not stated in the notes.\n"
"- When summarizing, include only the most relevant information related to the IBD diagnosis.\n"
"- Use direct quotes from the notes wherever possible to support your answer.";


std::string crc_system = 
"The text provided is a pathology report, with samples originating from the colon or rectum unless specified otherwise."
" We are interested in identifying whether invasive adenocarcinoma (stage greater than or equal to 1) is present in *any* colon or rectal sample."
" Without definite invasion identified, conditions such as 'high-grade dysplasia', 'in-situ [adeno]carcinoma', or 'intramucosal [adeno]carcinoma' are not typically classified as invasive adenocarcinoma."
" If the sample is classified as having adenocarcinoma without further specification, this typically implies invasive adenocarcinoma."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?";

std::string siteStage_system =
"The text provided is a pathology report diagnosing adenocarcinoma in the colon or rectum (CRC)."
" Identify the site of the cancer."// or the distance from the anal verge, in cm."
" Also label the TNM stage of the colorectal adenocarcinoma, if the stage is explicitly stated."
" If the stage of the cancer is not clear, label it as Unknown." //Alternatively, provide the minimum stage, such as 1+, 2+, or 3+."
" If the site it not clear, label it as Unknown or Unspecified colon."
" Format your answer as follows:\n"
"Site: {Primary tumor site or distance from anal verge}\n"
"Stage: {TNM stage, Not Invasive or Unknown}";

std::string advNeo_system = 
"The text provided is a pathology report, with samples originating from the colon or rectum unless specified otherwise."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that the patient has"
" adenocarcinoma or high-grade dysplasia"
" in any colon or rectal sample?";

std::string lgd_system = 
"The text provided is a pathology report."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that the patient has"
" any type of adenoma, adenomatous/dysplastic lesion(s), or dysplasia of any grade"
" in any colon or rectal sample? Exclude sessile serrated adenomas unless they are specified to have dysplasia.";

std::string ind_system = 
"The text provided is a pathology report."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report describe findings consistent with 'indefinite dysplasia'?"
" Exclude clear diagnoses of dysplasia (low-grade or high-grade) and non-dysplastic changes (e.g., inflammatory changes without atypia)."
" Focus specifically on cases where the pathologist explicitly indicates uncertainty or borderline features, often requiring follow-up or further sampling.";

std::string ind_system = "The text provided is a pathology report."
" Answer yes or no to the following question, matching the format 'Answer: Yes' or 'Answer: No'. Then, explain your reasoning."
" Does the pathology report indicate that any colon or rectal sample contains tissue which is considered 'indefinite for dysplasia'?"
" 'Indefinite for dysplasia' is defined as uncertainty about the presence of dysplasia and may include descriptions of equivocal findings"
" which are insufficient to confirm or exclude dysplasia. Answer yes if any colon or rectal sample is considered indefinite for dysplasia.";

std::string lgdClass_system = R"(The text provided is a pathology report diagnosing dysplasia or adenoma.
Classify each adenoma or dysplastic lesion in JSON format with the following fields: "description" "lesion_type", "sample_ID", "indication", "location", "size_mm", "shape", "dysplasia_grade", "background_inflammation", and "multifocal".
Choose a "lesion_type" based on the following list: "tubular adenoma", "sessile serrated adenoma", "traditional serrated adenoma", "tubulovillous adenoma", "villous adenoma", "villotubular adenoma", "low grade dysplasia", "high grade dysplasia", "inflammation", "dysplasia", "polyp with dysplasia", "post-inflammatory polyp with dysplasia", "adenomatous polyp", "sessile serrated polyp with dysplasia", "indefinite dysplasia", "dysplasia-associated lesion or mass", or "indeterminate dysplasia".
If "lesion_type" cannot be classified using the given list, fill each field with "NULL".
For "size_mm", write the length of the largest dimension and make sure the output is in millimeters. The size information is usually found in the Gross Description section of the report.
Choose a "shape" based on the following list: "pedunculated", "sessile", "flat", "flat elevated", "flat depressed", "invisible", "mass", "polypoid", or "nonpolypoid". 
ONLY CLASSIFY THE SAMPLES THAT ARE ADENOMA AND/OR DYSPLASIA IN THE COLON OR RECTUM!

Format each entry as follows:

[
  {
    "description": "<description of adenoma/dysplasia or NULL>",
    "lesion_type": "<type of lesion or NULL>",
    "sample_ID": "<letter or number identifying the adenoma or dysplastic lesion>",
    "indication": "<why was the biopsy taken>",
    "location": "<location of lesion>",
    "size_mm": "<size in mm or null if not applicable>",
    "shape": "<shape of lesion>",
    "dysplasia_grade": "<grade of dysplasia>",
    "background_inflammation": "<description of background colitis or inflammation>"
    "multifocal": "<are multiple adenomas or dysplasias described at once? yes, no, or unknown>"
  }
]

Example input:
- A. Ascending colon polyps, tubular adenomas (x2), each measuring 0.5 cm x 0.3 cm x 0.3 cm, tubular architecture, smooth surface, tan color, no background colitis or inflammation.
- #3. Low-grade dysplasia found on random biopsy from descending colon, not visible on endoscopy, with mildly inflamed background mucosa.

Expected output:
[
  {
    "description": "2 tubular adenomas in ascending colon",
    "lesion_type": "tubular adenoma",
    "sample_ID": "A",
    "indication": "polyp",
    "location": "ascending colon",
    "size_mm": "5",
    "shape": "polypoid",
    "dysplasia_grade": "low grade",
    "background_inflammation": "no inflammation",
    "multifocal": "yes, x2"
  },
  {
    "description": "invisible dysplasia from a random biopsy of the descending colon with background inflammation",
    "lesion_type": "invisible dysplasia",
    "sample_ID": "3",
    "indication": "random biopsy",
    "location": "descending colon",
    "size_mm": "null",
    "shape": "invisible",
    "dysplasia_grade": "low grade",
    "background_inflammation": "mild inflammation",
    "multifocal": "unknown"
  }
])";



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
    if (extractionDx == "ibd"){
        postSys = "Excerpts:\n";
    } else if (extractionDx == "crc" || extractionDx == "advNeo" || extractionDx == "lgd" || extractionDx == "ind" || 
            extractionDx == "siteStage" || extractionDx == "lgdClass"){
        postSys = "<<<\nPathology report:\n";
    } else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, ibd, advNeo, lgd, ind.");
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

std::string ibd_question = "";
std::string crc_question = ">>>\n\nDoes the pathology report indicate that the patient has an invasive adenocarcinoma in any colon or rectal sample?";
std::string siteStage_question = ">>>";
std::string lgdClass_question = ">>>";
std::string advNeo_question = ">>>\n\nDoes the pathology report indicate that the patient has"
" adenocarcinoma or high-grade dysplasia"
" in the colon or rectum?";
std::string lgd_question = ">>>\n\nDoes the pathology report indicate that the patient has"
" any type of adenoma (excluding sessile serrated adenoma), adenomatous/dysplastic lesion(s), or dysplasia of any grade"
" in the colon or rectum?";
std::string ind_question = ">>>\n\nDoes the pathology report indicate that the patient has"
" any colon or rectal sample that is indefinite for dysplasia?";

std::string ibd_preAnswer = "Summary from notes:";
std::string yesNo_preAnswer = "Answer:";
std::string siteStage_preAnswer = "Site:";
std::string lgdClass_preAnswer = "[";

std::string generatePreAnswer(const std::string& promptFormat, const std::string& extractionDx) {

    std::string question;
    std::string preAnswer;

    if (extractionDx == "ibd"){
        question = ibd_question;
        preAnswer = ibd_preAnswer;
    } else if (extractionDx == "crc"){
        question = crc_question;
        preAnswer = yesNo_preAnswer;
    } else if(extractionDx == "advNeo"){
        question = advNeo_question;
        preAnswer = yesNo_preAnswer;
    } else if(extractionDx == "lgd"){
        question = lgd_question;
        preAnswer = yesNo_preAnswer;
    } else if(extractionDx == "ind"){
        question = ind_question;
        preAnswer = yesNo_preAnswer;
    } else if(extractionDx == "siteStage" || extractionDx == "siteAN"){
        question = siteStage_question;
        preAnswer = siteStage_preAnswer;
    } else if(extractionDx == "lgdClass"){
        question = lgdClass_question;
        preAnswer = lgdClass_preAnswer;
    } else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, ibd, advNeo, lgd, ind.");
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
        if (smpl) {
            common_sampler_free(smpl);
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

    struct common_sampler * smpl = nullptr;
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

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL)) {
        return 1;
    }

    common_init();
    
    std::string system;
    if (params.extractionType == "ibd"){
        system = ibd_system;
    } else if (params.extractionType == "crc"){
        system = crc_system;
    } else if(params.extractionType == "advNeo"){
        system = advNeo_system;
    } else if(params.extractionType == "lgd"){
        system = lgd_system;
    } else if(params.extractionType == "ind"){
        system = ind_system;
    } else if(params.extractionType == "siteStage"){
        system = siteStage_system;
    } else if(params.extractionType == "lgdClass"){
        system = lgdClass_system;
    } else{
        throw std::runtime_error("Error: extraction type not recognized. Recgonized options are: crc, ibd, advNeo, lgd, ind.");
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

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model 

    // load the target model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    //llama_model * model = NULL;
    //llama_context * ctx = NULL;

    // initialize the context

    //llama_context_params ctx_params = llama_context_params_from_common_params(params);

    //llama_context * ctx = llama_new_context_with_model(model, ctx_params);

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
    tokens_format = ::common_tokenize(ctx, promptFormat_example, true);
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
        client.smpl = common_sampler_init(model, params.sampling);
    }

    // Initialize system prompt token vec
    std::vector<llama_token> tokens_system;
    // Format system prompt
    std::string k_system = formatSystemPrompt(system, params.promptFormat, params.extractionType);
    // Print the string
    printf("System prompt: %s\n", k_system.c_str());
    tokens_system = common_tokenize(ctx, k_system, true);
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

    LOG_INF("%s: Simulating parallel requests from %d patients:\n", __func__, n_seq);
    LOG_INF("%s: n_parallel (number of simultaneous requests) = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, cont_batching, n_tokens_system);
    LOG_INF("\n");

    {
        LOG_INF("%s: Evaluating the system prompt ...\n", __func__);

        for (int32_t i = 0; i < n_tokens_system; ++i) {
            common_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i <= n_clients; ++i) {
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }

        LOG_INF("\n");
    }

    LOG_INF("Processing requests ...\n\n");

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
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        common_batch_clear(batch);

        // decode any currently ongoing sequences
        for (auto & client : clients) {
            if (client.seq_id == -1) {
                continue;
            }

            client.i_batch = batch.n_tokens;

            common_batch_add(batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, { client.id + 1 }, true);

            client.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 1; i <= n_clients; ++i) {
                llama_kv_cache_seq_rm(ctx, i, -1, -1);
                // but keep the system prompt
                llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
            }

            //LOG_INF("%s: clearing the KV cache\n", __func__);
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

                    common_sampler_reset(client.smpl);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::common_tokenize(ctx, client.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        common_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { client.id + 1 }, false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.i_batch   = batch.n_tokens - 1;

                    if(params.patient_file.empty()){
                        LOG_INF("\n\n\033[0mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);
                    }else{
                        LOG_INF("\n\n\033[0mClient %3d, Patient %s, seq %4d, started decoding ...\033[0m\n", client.id, client.ptID.c_str(), client.seq_id);
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
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_ERR("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG_ERR("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG_DBG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = common_sampler_sample(client.smpl, ctx, client.i_batch - i);

                common_sampler_accept(client.smpl, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = common_token_to_piece(ctx, id);

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
                    const std::string eos_str = common_token_to_piece(ctx, llama_token_eos(model));
                    int32_t eot_token = llama_token_eot(model);

                    size_t pos;
                    if (eot_token == -1) {
                        pos = client.response.rfind(eos_str);
                    } else{
                        const std::string eot_str = common_token_to_piece(ctx, llama_token_eot(model));
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
                    if(!client.ptID.empty() && params.extractionType != "lgdClass"){
                        outFile3 << escapeNewLines(client.response);
                        outFile3 << "\t" << client.ptID;
                    } else if (params.extractionType == "lgdClass" && !client.ptID.empty()){
                        outFile3 << addFieldToJsonArray(client.response, "SurgPathSID", client.ptID);
                    } else{
                        std::cerr << "No Patient ID or SurgPathSID to identify each input!" << std::endl;
                        return 1;
                    }
                    outFile3 << std::endl;

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_INF("\033[0m \nInput:\n\033[96m%s\033[91m%s\033[0m\n\033[92mJust completed: Patient: %s, sequence %3d of %3d, prompt: %4d tokens, response: %4d tokens, time: %5.2f seconds, speed %5.2f t/s",
                            //escapeNewLines(client.input).c_str(),
                            client.input.c_str(),
                            client.response.c_str(),
                            client.ptID.c_str(), client.seq_id, n_seq, client.n_prompt, client.n_decoded,
                            (t_main_end - client.t_start_prompt) / 1e6,
                            (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6);
                            // n_cache_miss,
                            //k_system.c_str(),
                            //::trim(prompts[promptNumber]).c_str());
                    
                    LOG_INF("\nJust completed Patient: %s",
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

    LOG_INF("\n%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_INF("External prompt file: \033[32m%s\033[0m\n", params.prompt_file.c_str());
    LOG_INF("Model and path used:  \033[32m%s\033[0m\n\n", params.model.c_str());

    LOG_INF("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Cache misses:        %6d\n", n_cache_miss);

    LOG_INF("\n");

    //llama_print_timings(ctx);
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
