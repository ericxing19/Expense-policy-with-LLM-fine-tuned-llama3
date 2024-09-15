import re
import numpy as np 
import random
import pandas as pd
from datasets import Dataset

# Generate rejected
def generate_rejected(data_frame):
    """
    pre-process the dataset to generate rejected answer from other prompts
    """
    random_seed = 202465
    random.seed(random_seed)
    rejected_T1 = []
    rejected_Amount = []
    rejected_T2 = []
    # rejected_reason = []

    for i in range(0,len(data_frame)):
        change_made = False
        counter = 0
        while not change_made:
            
            binary_change = bin(random.randint(1, 15))[2:].zfill(4)
            counter += 1
            rand_index = random.randint(0, len(data_frame)-1)
            if binary_change[0] == '1':
                T1 = data_frame['Classification T1'][rand_index]
            else:
                T1 = data_frame['Classification T1'][i]
            if binary_change[1] == '1':            
                Am = data_frame['Amount'][rand_index]
            else:
                Am = data_frame['Amount'][i]
            if binary_change[2] == '1':
                T2 = data_frame['Classification T2'][rand_index]
            else:
                T2 = data_frame['Classification T2'][i]
            # if binary_change[3] == '1':
            #     Re = data_frame['Reasons'][rand_index]...

            if (T1 != data_frame['Classification T1'][i]) and (Am != data_frame['Amount'][i]) and (T2 != data_frame['Classification T2'][i]):
                change_made = True
            elif counter > 10:
                print(f"generate failed for prompt {i}")

        # now that changed are made:
        rejected_T1.append(T1)
        rejected_Amount.append(Am)
        rejected_T2.append(T2)
        # rejected_Re.append(Re)

    rejected_df = pd.DataFrame({
        'rejected_T1': rejected_T1,
        'rejected_Amount': rejected_Amount,
        'rejected_T2': rejected_T2,
        # 'rejected_reasons': rejected_Re,
    })

    # Concatenate the original DataFrame with the rejected DataFrame
    data_combine = pd.concat([data_frame, rejected_df], axis=1)
    return data_combine


# Generate rejected
# only one label change (use or)
def generate_rejected_one(data_frame):
    """
    pre-process the dataset to generate rejected answer from other prompts
    """
    random_seed = 202465
    random.seed(random_seed)
    rejected_T1 = []
    rejected_Amount = []
    rejected_T2 = []
    # rejected_reason = []

    for i in range(0,len(data_frame)):
        change_made = False
        counter = 0
        while not change_made:
            
            binary_change = bin(random.randint(1, 15))[2:].zfill(4)
            counter += 1
            rand_index = random.randint(0, len(data_frame)-1)
            if binary_change[0] == '1':
                T1 = data_frame['Classification T1'][rand_index]
            else:
                T1 = data_frame['Classification T1'][i]
            if binary_change[1] == '1':            
                Am = data_frame['Amount'][rand_index]
            else:
                Am = data_frame['Amount'][i]
            if binary_change[2] == '1':
                T2 = data_frame['Classification T2'][rand_index]
            else:
                T2 = data_frame['Classification T2'][i]
            # if binary_change[3] == '1':
            #     Re = data_frame['Reasons'][rand_index]...

            if (T1 != data_frame['Classification T1'][i]) or (Am != data_frame['Amount'][i]) or (T2 != data_frame['Classification T2'][i]):
                change_made = True
            elif counter > 10:
                print(f"generate failed for prompt {i}")

        # now that changed are made:
        rejected_T1.append(T1)
        rejected_Amount.append(Am)
        rejected_T2.append(T2)
        # rejected_Re.append(Re)

    rejected_df = pd.DataFrame({
        'rejected_T1': rejected_T1,
        'rejected_Amount': rejected_Amount,
        'rejected_T2': rejected_T2,
        # 'rejected_reasons': rejected_Re,
    })

    # Concatenate the original DataFrame with the rejected DataFrame
    data_combine = pd.concat([data_frame, rejected_df], axis=1)
    return data_combine

    
# Generate rejected T1
def generate_rejected_T1(data_frame):
    """
    pre-process the dataset to generate rejected answer from other prompts
    """
    random_seed = 202465
    random.seed(random_seed)
    rejected_T1 = []
    # rejected_reason = []

    for i in range(0,len(data_frame)):
        change_made = False
        counter = 0
        while not change_made:
            
            binary_change = bin(random.randint(1, 15))[2:].zfill(4)
            counter += 1
            rand_index = random.randint(0, len(data_frame)-1)
            if binary_change[0] == '1':
                T1 = data_frame['Classification T1'][rand_index]
            else:
                T1 = data_frame['Classification T1'][i]

            if (T1 != data_frame['Classification T1'][i]):
                change_made = True
            elif counter > 10:
                pass
                # print(f"generate failed for prompt {i}")

        # now that changed are made:
        rejected_T1.append(T1)
        # rejected_Re.append(Re)

    rejected_df = pd.DataFrame({
        'rejected_T1': rejected_T1,
        # 'rejected_reasons': rejected_Re,
    })

    # Concatenate the original DataFrame with the rejected DataFrame
    data_combine = pd.concat([data_frame, rejected_df], axis=1)
    return data_combine


# create dataset
def data_format_T1(example, context):
    # Format system
    system = f"""A <<<POLICY>>> and a <<<SCENARIO>>> will be provided, you are GREAT at analysing the <<<SCENARIO>>> according to the <<<POLICY>>>. Your judgment should strictly follow the policy and not add subjective judgment. \n Your answer MUST and and CAN ONLY be a valid json format, having 2 text fields: Classsifcation T1 and Reasons. Your answer should follow these structures: \n 1) "Classsifcation T1" :you MUST choose the activity in the <<<scenario>>> is 'Policy Violated' or 'Policy not Violated'\n If one of the policies is violated in the scenario, this scenario should be seen as "Policy Violated". \n 2) give reasons for your choices according to the <<<POLICY>>>.\n
    <<<<<Policy Start:{context}
    Policy End>>>>>"""

    # Format instruction
    prompt = f"""<<<<<Scenario Start:\n\n{example['Prompt']}\n\nScenario End>>>>>"""
    
    # Format chosen answer
    chosen = f"""Answer: {{"Classification T1": {example['Classification T1']},
    "Reasons": ...}} 
    <eos>\n"""

    # Format rejected answer
    rejected = f"""Answer: {{"Classification T1": {example['rejected_T1']},
    "Reasons": ...}}
    <eos>\n"""

    return {
        "prompt": system + "\n" + prompt,
        "chosen": chosen,
        # "rejected":"",
        "rejected":rejected,
    }


def data_format(example, context):
    # Format system
    system = f"""A <<<POLICY>>> and a <<<SCENARIO>>> will be provided, you are GREAT at analysing the <<<SCENARIO>>> according to the <<<POLICY>>>. Your answer MUST and and CAN ONLY be a valid json format, having 4 text fields: Classification T1, Reimbursement Amount, Classification T2, and Reasons. Your answer should follow these structures: \n 1) "Classification T1" :you MUST choose the activity in the <<<scenario>>> is 'Policy Violated' or 'Policy not Violated'\n \n 2) "Reimbursement Amount" : Answer according to the Policy the amount of money can be reimbursed totally \n 3) "Classification T2" : Compare to the requested reimbursement amount in the scenario, and answer if the claim is 'Fully Reimbursable'/'Partially Reimbursable'/'Not Reimbursable'/'Further Clarification Required'. \n 4) give reasons for your choices according to the <<<POLICY>>>.
    <<<<<Policy Start:{context}
    \n\n\nPolicy End>>>>>\n\n"""

    # Format instruction
    prompt = f"""<<<<<Scenario Start:\n\n{example['Prompt']}\n\nScenario End>>>>>"""

    # Format chosen answer
    chosen = f"""Answer: {{"Classification T1": {example['Classification T1']},
    "Reimbursement Amount": £{example['Amount']},
    "Classifcation T2": {example['Classification T2']},
    "Reasons": ...}} 
    <eos>\n"""

    # Format rejected answer
    rejected = f"""Answer: {{"Classification T1": {example['rejected_T1']},
    "Reimbursement Amount": £{example['rejected_Amount']},
    "Classifcation T2": {example['rejected_T2']},}} 
    <eos>\n"""

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        # "rejected":"",
        "rejected":rejected,
    }
    
def data_format_category(example, context_list):
    category = example['Category']
    # Format system
    system = f"""A <<<POLICY>>> and a <<<SCENARIO>>> will be provided, you are GREAT at analysing the <<<SCENARIO>>> according to the <<<POLICY>>>. Your answer MUST and and CAN ONLY be a valid json format, having 4 text fields: Classification T1, Reimbursement Amount, Classification T2, and Reasons. Your answer should follow these structures: \n 1) "Classification T1" :you MUST choose the activity in the <<<scenario>>> is 'Policy Violated' or 'Policy not Violated'\n \n 2) "Reimbursement Amount" : Answer according to the Policy the amount of money can be reimbursed totally \n 3) "Classification T2" : Compare to the requested reimbursement amount in the scenario, and answer if the claim is 'Fully Reimbursable'/'Partially Reimbursable'/'Not Reimbursable'/'Further Clarification Required'. \n 4) give reasons for your choices according to the <<<POLICY>>>.
    <<<<<Policy Start:{context_list[category]}
    \n\n\nPolicy End>>>>>\n\n"""

    # Format instruction
    prompt = f"""<<<<<Scenario Start:\n\n{example['Prompt']}\n\nScenario End>>>>>"""

    # Format chosen answer
    chosen = f"""Answer: {{"Classification T1": {example['Classification T1']},
    "Reimbursement Amount": £{example['Amount']},
    "Classifcation T2": {example['Classification T2']},
    "Reasons": ...}} 
    <eos>\n"""

    # Format rejected answer
    rejected = f"""Answer: {{"Classification T1": {example['rejected_T1']},
    "Reimbursement Amount": £{example['rejected_Amount']},
    "Classifcation T2": {example['rejected_T2']},}} 
    <eos>\n"""

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        # "rejected":"",
        "rejected":rejected,
    }

# Only process t1 label when label = 't1', precess all labels when label = 'full'
def create_dataset(data_frame, context, label):

    if(label == 't1'):
        temp = generate_rejected_T1(data_frame)
        # Convert the pandas DataFrame to a Hugging Face Dataset
        tr_data = Dataset.from_pandas(temp)
        # Save columns
        original_columns = tr_data.column_names
        # Format dataset
        tr_dataset = tr_data.map(
            data_format_T1,
            fn_kwargs={"context": context},  # Pass the context argument here
            remove_columns=original_columns # remove un-used columns for training process
        )
    else:
        temp = generate_rejected(data_frame)
        # Convert the pandas DataFrame to a Hugging Face Dataset
        tr_data = Dataset.from_pandas(temp)
        # Save columns
        original_columns = tr_data.column_names
        # Format dataset
        tr_dataset = tr_data.map(
            data_format,
            fn_kwargs={"context": context},  # Pass the context argument here
            remove_columns=original_columns # remove un-used columns for training process
        )


    print("Here's an example of train data: ")
    print("Length of prompt: ", len(tr_dataset[5]['prompt']))
    print("Scenario: ", data_frame['Prompt'][5])
    print("Chosen: ", tr_dataset[5]['chosen'])
    print("Rejected: ", tr_dataset[5]['rejected'])
    
    # acc_test_frame['Reasons'] = acc_test_frame['GPT 4 Chosen Response']
    # print(acc_test_frame)
    # if(label == 't1'):
    #     test_temp = generate_rejected_T1(test_frame)
    #     # Convert the pandas DataFrame to a Hugging Face Dataset
    #     test_data = Dataset.from_pandas(test_temp)
    #     # Save columns
    #     test_columns = test_data.column_names
    #     # Format dataset
    #     test_dataset = test_data.map(
    #         data_format_T1,
    #         fn_kwargs={"context": context},  # Pass the context argument here
    #         remove_columns=test_columns # remove un-used columns for training process
    #     )
    # else:
    #     test_temp = generate_rejected(test_frame)
    #     # Convert the pandas DataFrame to a Hugging Face Dataset
    #     test_data = Dataset.from_pandas(test_temp)
    #     # Save columns
    #     test_columns = test_data.column_names
    #     # Format dataset
    #     test_dataset = test_data.map(
    #         data_format,
    #         fn_kwargs={"context": context},  # Pass the context argument here
    #         remove_columns=test_columns # remove un-used columns for training process
    #     )

    # print("Here's an example of test data: ")
    # print(data_frame['Prompt'][5])
    # print(test_dataset[5]['chosen'])
    # print(test_dataset[5]['rejected'])
    
    return tr_dataset

# Only process t1 label when label = 't1', precess all labels when label = 'full'
def create_dataset_category(data_frame, context_list, label):
    if(label == 't1'):
        temp = generate_rejected_T1(data_frame)
        # Convert the pandas DataFrame to a Hugging Face Dataset
        tr_data = Dataset.from_pandas(temp)
        # Save columns
        original_columns = tr_data.column_names
        # Format dataset
        tr_dataset = tr_data.map(
            data_format_category, # here T1 is useless, all label are full!!!!!
            fn_kwargs={"context_list": context_list},  # Pass the context argument here
            remove_columns=original_columns # remove un-used columns for training process
        )
    else:
        temp = generate_rejected(data_frame)
        # Convert the pandas DataFrame to a Hugging Face Dataset
        tr_data = Dataset.from_pandas(temp)
        # Save columns
        original_columns = tr_data.column_names
        # Format dataset
        tr_dataset = tr_data.map(
            data_format_category,
            fn_kwargs={"context_list": context_list},  # Pass the context argument here
            remove_columns=original_columns # remove un-used columns for training process
        )


    print("Here's an example of train data: ")
    print("Length of prompt: ", len(tr_dataset[5]['prompt']))
    print("Scenario: ", data_frame['Prompt'][5])
    print("Chosen: ", tr_dataset[5]['chosen'])
    print("Rejected: ", tr_dataset[5]['rejected'])
    
    return tr_dataset



####################
# update train dataset with new generated rejected response
####################

def replace_rejected(example, current_idx, replace_idx, re_context):
    # Check if the current index is in the replace_idx list
    if current_idx in replace_idx:
        # Replace the 'rejected' field in the example with the corresponding value from re_context
        example['rejected'] = re_context[current_idx]
    return example

def update_rejected_responses(tr_dataset, replace_idx, replace_context):
    updated_dataset = tr_dataset.map(
        replace_rejected,
        with_indices=True,  # Enables passing the current sample's index to the replace_rejected function
        fn_kwargs={'replace_idx': replace_idx, 're_context': replace_context}
    )
    return updated_dataset

def filter_by_rejected_length(example):
    # Return the example only if the length of the 'rejected' field is greater than 200
    return len(example['rejected']) > 200

def filter_tr_dataset(tr_dataset):
    # Filter the dataset based on the condition defined in the filter_by_rejected_length function
    filtered_dataset = tr_dataset.filter(filter_by_rejected_length)
    return filtered_dataset

def extract_key_values(text):
    """
    Extracts the values of "Classification T1", "Reimbursement Amount", and "Classification T2" from the given text.
    """
    pattern_t1 = re.search(r'"Classification T1"\s*:\s*"([^"]+)"', text)
    pattern_amount = re.search(r'"Reimbursement Amount"\s*:\s*([£\d.]+)', text)
    pattern_t2 = re.search(r'"Classsification T2"\s*:\s*"([^"]+)"', text)

    classification_t1 = pattern_t1.group(1) if pattern_t1 else None
    reimbursement_amount = pattern_amount.group(1) if pattern_amount else None
    classification_t2 = pattern_t2.group(1) if pattern_t2 else None

    return classification_t1, reimbursement_amount, classification_t2

def filter_identical_samples(example):
    # Extract key values from both 'chosen' and 'rejected' fields
    chosen_values = extract_key_values(example['chosen'])
    rejected_values = extract_key_values(example['rejected'])

    # If all three fields are identical, return False (indicating the sample should be removed), otherwise return True
    return chosen_values != rejected_values

def filter_tr_dataset_A(tr_dataset):
    # Use the filter function to remove samples with identical key values in 'chosen' and 'rejected'
    filtered_dataset = tr_dataset.filter(filter_identical_samples)
    return filtered_dataset

def correct_typo(example):
    # Correct the typo in both 'chosen' and 'rejected' fields
    example['chosen'] = example['chosen'].replace("Classsifcation T2", "Classification T2")
    example['rejected'] = example['rejected'].replace("Classsifcation T2", "Classification T2")
    return example

def correct_dataset_typo(tr_dataset):
    # Apply the typo correction to the entire dataset
    corrected_dataset = tr_dataset.map(correct_typo)
    return corrected_dataset