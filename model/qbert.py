import torch
import torch.nn as nn
from transformers import BertModel

class QsingBertModel(nn.Module):
    def __init__(self):
        super(QsingBertModel, self).__init__()
        
        self.bert_urls = BertModel.from_pretrained('bert-base-uncased')
        self.bert_html = BertModel.from_pretrained('bert-base-uncased')
        
        self.fc = nn.Linear(768 * 2, 512)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        url_input_ids = input['url_input_ids']
        url_attention_mask = input['url_attention_mask']
        html_input_ids = input['html_input_ids']
        html_attention_mask = input['html_attention_mask']
        
        url_output = self.bert_urls(input_ids=url_input_ids, attention_mask=url_attention_mask)
        url_cls_embedding = url_output.last_hidden_state[:, 0, :]
        
        html_output = self.bert_html(input_ids=html_input_ids, attention_mask=html_attention_mask)
        html_cls_embedding = html_output.last_hidden_state[:, 0, :]

        combined = torch.cat((url_cls_embedding, html_cls_embedding), dim=1)

        x = self.fc(combined)
        x = self.gelu(x)

        logits = self.output_layer(x)
        logits = logits.squeeze()
        output = self.sigmoid(logits)

        return logits, output
