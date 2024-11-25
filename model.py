import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
torch.manual_seed(42)

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        
    def forward(self, query, key, value):
        # Project query, key, and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        # Compute attention
        output, _ = self.attention(query, key, value)
        return output

class PVRModel(nn.Module):
    def __init__(self, d_model, model_name):
        super(PVRModel, self).__init__()
        self.cross_attn_rules = CrossAttentionLayer(d_model)
        self.cross_attn_facts = CrossAttentionLayer(d_model)
        
        # Learnable weights for combining z1 and z2
        self.lambda1 = nn.Parameter(torch.tensor(0.5))  # Initialized to 0.5
        self.lambda2 = nn.Parameter(torch.tensor(0.5))  # Initialized to 0.5
        
        # RoBERTa model for final processing
        self.roberta = RobertaModel.from_pretrained(model_name)
        for param in self.roberta.parameters():
            param.requires_grad = False  # Freeze RoBERTa model
        
        # Linear layer for final output
        self.fc = nn.Linear(d_model, 2)  # Assuming binary classification (adjust as needed)

    def forward(self, question_ids, rules_ids, facts_ids, labels=None):
        """
        Args:
            question_ids: Token IDs for questions (batch_size, seq_len)
            rules_ids: Token IDs for rules (batch_size, seq_len)
            facts_ids: Token IDs for facts (batch_size, seq_len)
            attention_mask: Optional attention mask for token padding (batch_size, seq_len)
        """
        # Convert token IDs to embeddings using RoBERTa's embedding layer
        question_embeds = self.roberta.embeddings(input_ids=question_ids)
        rules_embeds = self.roberta.embeddings(input_ids=rules_ids)
        facts_embeds = self.roberta.embeddings(input_ids=facts_ids)
        
        # Cross-attention for rules
        z1 = self.cross_attn_rules(query=question_embeds, key=rules_embeds, value=rules_embeds)
        
        # Cross-attention for facts
        z2 = self.cross_attn_facts(query=question_embeds, key=facts_embeds, value=facts_embeds)
        
        # Weighted combination of z1 and z2
        z = self.lambda1 * z1 + self.lambda2 * z2
        
        # Pass combined representation through frozen RoBERTa layers
        roberta_output = self.roberta(inputs_embeds=z).last_hidden_state  # (batch_size, seq_len, d_model)
        
        # Final classification using the [CLS] token representation
        res = self.fc(roberta_output[:, 0, :])  # Use [CLS] token's representation
        
        # Calculate loss
        if labels is not None:
            loss = F.cross_entropy(input=res, target=labels)
        else:
            loss = None
        return {"logits": res, "loss": loss}
    
    def calc_num_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"Training efficiency: {train_params * 100 / total_params:.3f}%")

def main():
    # Example dimensions
    batch_size = 8
    seq_len = 512
    d_model = 1024

    # Random tensors for question, rules, and facts
    question = torch.randint(size=(batch_size, seq_len), low=1, high=1000).to("cuda")
    rules = torch.randint(size=(batch_size, seq_len), low=1, high=1000).to("cuda")
    facts = torch.randint(size=(batch_size, seq_len), low=1, high=1000).to("cuda")
    labels = torch.randint(size=(batch_size,), low=0, high=2).to("cuda")
    
    # Instantiate the model
    model = PVRModel(d_model, model_name="LIAMF-USP/roberta-large-finetuned-race").to("cuda")

    # Forward pass
    output = model(question, rules, facts, labels=labels)
    print(output)  # Should be (batch_size, 2) for binary classification

if __name__ == "__main__":
    main()
    