import json, torch, tqdm
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

class LogicDataset(Dataset):
    def __init__(self, filepath, model_name, frac):
        """
        Initialize the dataset by loading the JSON file and preparing data.
        Args:
            filepath (str): Path to the JSON file containing the dataset.
            model_name (str): Pretrained model name for the tokenizer.
        """
        with open(filepath, "r") as file:
            raw_data = [json.loads(line) for line in file]  # Load each line as a JSON object

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.examples = []
        raw_data = raw_data[:int(len(raw_data)*frac)]
        
        # Process each JSON object
        with tqdm.tqdm(iterable=raw_data, desc="Preparing data...", total=len(raw_data)) as pbar:
            for instance in pbar:
                # Extract facts and rules (shared across questions)
                facts = instance.get("triples", {})
                facts_text = " ".join([fact["text"] for fact in facts.values()])

                rules = instance.get("rules", {})
                rules_text = " ".join([rule["text"] for rule in rules.values()])

                # Tokenize facts and rules once (reuse across all questions)
                facts_tokens = self.tokenizer(
                    facts_text,
                    add_special_tokens=True,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"][0]
                rules_tokens = self.tokenizer(
                    rules_text,
                    add_special_tokens=True,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"][0]

                # Extract questions and create an example for each
                questions = instance.get("questions", {})
                for question_data in questions.values():
                    question_text = question_data["question"]
                    answer = int(bool(question_data["answer"]))
                    question_tokens = self.tokenizer(
                        question_text,
                        add_special_tokens=True,
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )["input_ids"][0]

                    # Append the example to the list
                    self.examples.append({
                        "question": question_tokens,
                        "facts": facts_tokens,
                        "rules": rules_tokens,
                        "answer": answer
                    })

    def __len__(self):
        """
        Return the number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        Args:
            idx (int): Index of the example to retrieve.
        Returns:
            dict: {"question": Tensor, "facts": Tensor, "rules": Tensor}
        """
        return self.examples[idx]

def main():
    # Path to the dataset and model name
    filepath = "data/depth-5/meta-train.jsonl"
    model_name = "roberta-base"

    # Create the dataset
    dataset = LogicDataset(filepath, model_name)

    # Create a DataLoader (optional)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # Iterate through the dataset
    for batch in dataloader:
        print(batch)
        break

if __name__ == "__main__":
    main()