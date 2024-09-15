import re
import numpy as np 
import random
import pandas as pd
import torch
from collections import Counter
import gc

def extract_labels(text):
    # label patterns
    # pattern1 = r"{.*?(Policy\sViolated|Policy\sNot\sViolated)"
    pattern1 = r'{[^{}]*?(Policy\sViolated|Policy\sNot\sViolated)'
    pattern2 = r"(\d+)\s*£|£\s*(\d+)|Reimbursement\sAmount.*?(\d+|0)"   # 3 groups to match the amount
    pattern3 = r"(Fully\s+Reimbursable|Partially\s+Reimbursable|Not\sReimbursable|Further\sClarification\sRequired)"
    
    # create match
    match1 = re.search(pattern1, text)
    match2 = re.search(pattern2, text)
    match3 = re.search(pattern3, text)
    
    # find match
    label1 = match1.group(1) if match1 else '-'
    # label2 = match2.group(1) if match2 else None
    label2 = -1
    if match2:
        label2 = match2.group(1) or match2.group(2) or match2.group(3)
        
    label3 = match3.group(1) if match3 else '-'
    
    # give a format marker , 1 if there's at least 1 classification label missing and 0 if not 
    label_format = 1 if label1 == '-' or label2 == -1 or label3 == '-' else 0 
    
    return label1, float(label2), label3, label_format

def extract_all_labels(scenario, category, response_list, num):
    label_list = []
    for i in range(num):
        text = response_list['original response'][i]
        label1, label2, label3, label_format = extract_labels(text)
        label_list.append([scenario[i], category[i], text, label1, label2, label3, label_format])
    return label_list

# Evaluate
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
def evaluate_label(response_df, prompt_df):
    T1_accurate = 0
    T2_accurate = 0
    amount_accurate = 0
    for i in range(len(response_df)):
        pre_label = response_df[['Classification T1', 'Amount', 'Classification T2','Format Compliance']].iloc[i]
        ori_label = prompt_df[['Classification T1', 'Amount', 'Classification T2']].iloc[i]

        if pre_label['Classification T1'] == ori_label['Classification T1']:
            T1_accurate += 1
        if float(pre_label['Amount']) == ori_label['Amount']:
            amount_accurate += 1
        if pre_label[ 'Classification T2'] == ori_label['Classification T2']:
           T2_accurate += 1
    T1_accuracy = T1_accurate/len(response_df) 
    T2_accuracy = T2_accurate/len(response_df)
    amount_accuracy = amount_accurate/len(response_df)
    format_accuracy = (1 - response_df['Format Compliance'].sum()/len(response_df)) * 100

    # F1 score calculation
    # To calculate F1 requires labelled inputs
    encoder = LabelEncoder()
    a = encoder.fit_transform(['Policy Violated', 'Policy Not Violated', '-'])
    f1_T1 = f1_score(encoder.transform(prompt_df['Classification T1'][:len(response_df)]), encoder.transform(response_df['Classification T1']), average='weighted')

    # print(a,encoder.transform(response_df['Classification T1']).sum())

    encoder.fit(['Fully Reimbursable','Partially Reimbursable','Not Reimbursable','Further Clarification Required','-'])
    #f1_amount = f1_score(prompt_df['Reimbursable Amount'], response_df['Reimbursement Amount'], average='weighted')
    f1_T2 = f1_score(encoder.transform(prompt_df['Classification T2'][:len(response_df)]), encoder.transform(response_df['Classification T2']), average='weighted')

    print(f"T1 acc: {T1_accuracy}, T1 F1 Score: {f1_T1};\nT2 acc: {T2_accuracy}, T2 weighted F1 Score: {f1_T2};\namount acc: {amount_accuracy}, \nformat acc: {format_accuracy} ")
    
    
def retrieve(retriever, query):
    results = retriever.get_relevant_documents(query)

    retrieve_context = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)])

    return retrieve_context 
    
def generate_multiple_responses_with_logits(model, tokenizer, prompt, num_responses, data_type):
    responses = []
    logits_scores = []
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs['input_ids'].shape[-1]  # get prompt length
    # print(inputs['input_ids'].dtype)
    # print(inputs['attention_mask'].dtype)
    with torch.no_grad():
        model.lm_head.weight.data = model.lm_head.weight.data.to(data_type)

    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}, dtype: {param.dtype}")
    
    for _ in range(num_responses):
        # Generate multiple responses from the model using beam search
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            min_new_tokens=2,
            repetition_penalty=2.0,
            num_beams=2,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # `outputs.sequences` shape: [batch_size * num_beams, sequence_length]
        # Select the best beam (usually the first beam in the output)
        best_beam_sequence = outputs.sequences[0, input_length:]
        
        # Decode the generated response for the best beam
        response = tokenizer.decode(best_beam_sequence, skip_special_tokens=True)
        responses.append(response)
        
        # Calculate logits scores for the best beam
        response_logits = torch.stack(outputs.scores, dim=1)  # Shape: [sequence_length, num_beams, vocab_size]
        log_probs = torch.log_softmax(response_logits[0], dim=-1)  # Calculate log softmax to obtain log probabilities
        # print("log_probs: ", log_probs.shape)
        
        # print("best_beam_sequence: ", best_beam_sequence.shape)
        # Ensure we gather logits for the correct sequences generated by the best beam
        selected_log_probs = log_probs.gather(1, best_beam_sequence.unsqueeze(-1)).squeeze(-1)  # Extract log probabilities for the actual generated tokens
        
        # Calculate the average log probability across the sequence for the best beam
        avg_log_prob = selected_log_probs.mean().item()
        logits_scores.append(avg_log_prob)
    
    del outputs, response_logits, log_probs, selected_log_probs
    
    gc.collect()
    
    torch.cuda.empty_cache()
    
    
    return responses, logits_scores

def select_preferred_and_rejected_based_on_logits(responses, logits_scores):
    # Select the preferred and rejected responses based on logits scores
    preferred_index = torch.argmax(torch.tensor(logits_scores)).item()  # Find the index of the highest score
    rejected_index = torch.argmin(torch.tensor(logits_scores)).item()  # Find the index of the lowest score
    
    preferred_response = responses[preferred_index]  # Get the response with the highest score
    rejected_response = responses[rejected_index]  # Get the response with the lowest score
    
    return preferred_response, rejected_response


# prompt_num: how many data points involved
def get_rejected_responses(model, tokenizer, prompt_num, context_list, data_frame, data_type = torch.bfloat16):
    rejected_response_list = []
    for i in range(prompt_num):
        category = data_frame['Category'][i]
        # iterating through all data points
        # prompt_all
        # Add question at the beginning
        prompt = f"""Question: A <<<POLICY>>> and a <<<SCENARIO>>> will be provided, you are GREAT at analysing the <<<SCENARIO>>> according to the <<<POLICY>>>. Your judgment should strictly follow the policy and not add subjective judgment. \n Your answer MUST and and CAN ONLY be a valid json format, having 4 text fields: Classification T1, Reimbursement Amount, Classification T2, and Reasons. Your answer should follow these structures: \n 1) "Classification T1":you MUST choose the activity in the <<<scenario>>> is 'Policy Violated' or 'Policy not Violated'\n  If one of the policies is violated in the scenario, this scenario should be seen as "Policy Violated". \n 2) "Reimbursement Amount" : Answer according to the Policy the amount of money can be reimbursed totally \n 3) "Classification T2" : Compare to the requested reimbursement amount in the scenario, and answer if the claim is 'Fully Reimbursable'/'Partially Reimbursable'/'Not Reimbursable'/'Further Clarification Required'. \n 4) give reasons for your choices according to the <<<POLICY>>>.\n\n Notice: The answer must start with the required json directly, don't add anything before the json answer (Don't repeat my question).

        <<<<<start examples:

        <<<<<example one:
        scenario: I used a TFL travel card to travel from home to the GNEI work station every day for a week and it cost me £50. How much can I reimburse?

        Answer:{{"Classification T1" : "Policy Violated",
                "Reimbursement Amount" : £0,
                "Classification T2" : "Not Reimbursable",
                "Reasons":"According to the GNEI Expenses Policy, travel between home and normal place of work (i.e., the GNEI campus) cannot be claimed. Thus, this expense cannot be reimbursed."}};

        <<<<<example two:
        scenario: I booked a taxi from the GNEI work station to my home yesterday night. The total cost was £25. I left the campus at 11:30pm. Can I expense this charge?

        Answer:{{"Classification T1" : "Policy Not Violated",
                "Reimbursement Amount" : £25,
                "Classification T2" : "Fully Reimbursable",
                "Reasons":"The GNEI Expenses Policy states that taxi fares can be claimed for journeys where a member of staff is working in the office very late, specifically after 11pm​​. Therefore, your taxi fare under these circumstances falls within the allowable expenses according to the policy, and total reimbursable amount is £25."}};
        
        <<<<<example three:
        scenario: I booked a taxi from the GNEI campus to my home yesterday night. The total cost was £35. I left the campus at 9:30pm. Can I expense this charge?

        Answer:{{"Classification T1" : "Policy Violated",
                "Reimbursement Amount" : £0,
                "Classification T2" : "Further Clarification Required",
                "Reasons":"Based on the GNEI Expenses Policy, taxi fares can only be claimed for specific reasons, such as if a member of staff is working in the office very late, specifically mentioned as after 11pm. Since you left the campus at 9:30pm, your taxi fare does not meet the criteria set out under the "Taxis" section for allowable taxi expenses.
                Therefore, based on the information provided in the GNEI Expenses Policy, you cannot expense the £35 taxi charge for a trip from the GNEI campus to your home that occurred at 9:30pm."}};
        
        <<<<<example four:
        scenario: I drove to Cambridge for a business trip along with another GNEI colleagues. The total distance covered was 60 miles. I accidentally hit the sidewalk and damaged the side of the car. The total cost of the repairs was £100. How much can I claim in expenses?  

        Answer:{{"Classification T1" : "Policy Violated",
                "Reimbursement Amount" : £30,
                "Classification T2" : "Partially Reimbursable",
                "Reasons":"For your business trip to Cambridge, driving a car for a total distance of 60 miles with another GNEI colleagues, you can claim expenses based on the GNEI Expenses Policy as follows:
                - **Mileage Claim**: According to section **4.32**, mileage for cars is reimbursed at a rate of 50p per mile for the first 11,000 miles in a tax year. Therefore, for 65 miles, the claim would be 60 miles x £0.50/mile = £30.
                - **Damage to the Vehicle**: Section **4.34** specifies that GNEI will not be held liable for damage or repairs to the vehicle used on GNEI’s business. Thus, the cost of repairs due to the accident, totaling £100, cannot be claimed under the policy.
                **Documentation and Approvals Required:**
                1. **Mileage Documentation**: Include details such as the start and end points of the journey, the purpose of the business trip, and the total number of miles claimed.
                2. **Approval for Business Travel**: Ensure that the business trip was pre-approved as per GNEI's travel policy requirements.
                3. **Vehicle Damage**: Since the policy excludes claims for vehicle damage or repairs, there is no requirement for documentation or approval in this context.

                **Total Amount That Can Be Expensed: £30** (for mileage only, as vehicle repair costs are not covered)."}};

        examples End>>>>>
        
        <<<<<Policy Start:{context_list[category]}
        \n\n\nPolicy End>>>>>\n\n<<<<<Scenario Start:\n\n{data_frame['Prompt'][i]}\n\nScenario End>>>>>"""
            
        # and show how much can be reimbursed\nScenario:\n{data_frame['Prompt'][i]}"""
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        # input = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        if(i % 10 == 0):
            print(input_ids['input_ids'].shape)
        
        
        responses, logits_scores = generate_multiple_responses_with_logits(model, tokenizer, prompt, num_responses=1, data_type = data_type)
        
        # print(logits_scores)
        
        preferred_response, rejected_response = select_preferred_and_rejected_based_on_logits(responses, logits_scores)

        # print("Preferred: ", preferred_response)
        # print("Rejected: ", rejected_response)

        # collecting responses
        # model_original_resp.append(trimmed_output)
        rejected_response_list.append(rejected_response)
        if(i % 10 == 0):
            print(f"the",(i+1),"th prompt responded")
        
    rejected_response_list = pd.DataFrame(rejected_response_list, columns=['original response'])
        
    return rejected_response_list



def evaluate_label_withidx(response_df, prompt_df):
    T1_accurate = 0
    T2_accurate = 0
    amount_accurate = 0
    rejected_idx = []
    for i in range(len(response_df)):
        pre_label = response_df[['Classification T1', 'Amount', 'Classification T2','Format Compliance']].iloc[i]
        ori_label = prompt_df[['Classification T1', 'Amount', 'Classification T2']].iloc[i]

        if pre_label['Classification T1'] == ori_label['Classification T1']:
            T1_accurate += 1
        else:
            rejected_idx.append(i)
        if float(pre_label['Amount']) == ori_label['Amount']:
            amount_accurate += 1
        else:
            rejected_idx.append(i)
        if pre_label['Classification T2'] == ori_label['Classification T2']:
           T2_accurate += 1
        else:
            rejected_idx.append(i)
        
    rejected_idx = list(dict.fromkeys(rejected_idx))
    print(rejected_idx)  
    
    T1_accuracy = T1_accurate/len(response_df) 
    T2_accuracy = T2_accurate/len(response_df)
    amount_accuracy = amount_accurate/len(response_df)
    format_accuracy = (1 - response_df['Format Compliance'].sum()/len(response_df)) * 100

    # F1 score calculation
    # To calculate F1 requires labelled inputs
    encoder = LabelEncoder()
    a = encoder.fit_transform(['Policy Violated', 'Policy Not Violated', '-'])
    f1_T1 = f1_score(encoder.transform(prompt_df['Classification T1'][:len(response_df)]), encoder.transform(response_df['Classification T1']), average='weighted')

    # print(a,encoder.transform(response_df['Classification T1']).sum())

    encoder.fit(['Fully Reimbursable','Partially Reimbursable','Not Reimbursable','Further Clarification Required','-'])
    #f1_amount = f1_score(prompt_df['Reimbursable Amount'], response_df['Reimbursement Amount'], average='weighted')
    f1_T2 = f1_score(encoder.transform(prompt_df['Classification T2'][:len(response_df)]), encoder.transform(response_df['Classification T2']), average='weighted')

    print(f"T1 acc: {T1_accuracy}, T1 F1 Score: {f1_T1};\nT2 acc: {T2_accuracy}, T2 weighted F1 Score: {f1_T2};\namount acc: {amount_accuracy}, \nformat acc: {format_accuracy} ")
    
    return rejected_idx