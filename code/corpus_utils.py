from datasets import load_dataset

def load_e2e_dataset():

    return load_dataset('GEM/e2e_nlg', trust_remote_code=True)  # loading the corpus for our study

def linearize_mr(mr: str, add_task_prefix: bool = True) -> str:
 
    #This function converts the  E2E NLG MR data meaning_representation string into a linearized text format tag = variable.

    slot_value_pairs = [pair.strip() for pair in mr.split(",")]
    formatted_pairs = []
    for pair in slot_value_pairs:
        if "[" in pair and "]" in pair:
            slot, value = pair.split("[", 1)
            slot = slot.strip()
            value = value.strip(" ]")
            formatted_pairs.append(f"{slot} = {value}")
        else:
            formatted_pairs.append(pair)
    linearized = " ; ".join(formatted_pairs)
    if add_task_prefix:
        return f"translate MR to text: {linearized}"
    return linearized



def simplify_train_dataset(batch):
    """
    this function removes the unnecessary information from the train dataset: and apply
    linearize MR function to the meaning_representation column. 
    """
    new_batch = {'graph': [], 'input': [], 'reference': []}
    for i in range(len(batch['meaning_representation'])):
        graph = batch['meaning_representation'][i]
        new_batch['graph'].append(graph)
        new_batch['input'].append(linearize_mr(graph))
        new_batch['reference'].append(batch['target'][i])
    return new_batch

def simplify_validation_dataset(batch):
    """
    this function removes the unnecessary information from the validation dataset: and apply
    linearize MR function to the meaning_representation column. 
    """
    new_batch = {'graph': [], 'input': [], 'reference': []}
    for i in range(len(batch['meaning_representation'])):
        graph = batch['meaning_representation'][i]
        new_batch['graph'].append(graph)
        new_batch['input'].append(linearize_mr(graph))
        new_batch['reference'].append(batch['references'][i][0])
    return new_batch

def simplify_test_dataset(batch):
    """
    this function removes the unnecessary information from the test dataset: and apply
    linearize MR function to the meaning_representation column. 
    """
    new_batch = {'graph': [], 'input': [], 'reference': []}
    for i in range(len(batch['meaning_representation'])):
        graph = batch['meaning_representation'][i]
        new_batch['graph'].append(graph)
        new_batch['input'].append(linearize_mr(graph))
        new_batch['reference'].append(batch['references'][i])
    return new_batch
