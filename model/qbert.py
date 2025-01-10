import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, MobileBertConfig, MobileBertModel


class UrlCnnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=768, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        
        # 예: 서로 다른 kernel_size를 쓰는 다중 Conv1D 예시
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        
        # 합쳐진 채널을 최종적으로 BERT와 같은 768 차원으로 맞춤
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            x = x * attention_mask.unsqueeze(-1)
        
        # 3) CNN에 입력하기 위해 (batch_size, embed_dim, seq_len)로 변환
        x = x.permute(0, 2, 1)
        
        x3 = self.relu(self.conv3(x))
        x5 = self.relu(self.conv5(x))
        
        # 5) Global Max Pooling → (batch_size, hidden_dim)
        x3_pool = torch.max(x3, dim=2).values
        x5_pool = torch.max(x5, dim=2).values
        
        # 6) 두 결과를 concat → (batch_size, hidden_dim*2)
        x_cat = torch.cat([x3_pool, x5_pool], dim=1)
        
        # 7) 원하는 출력 차원(예: 768)으로 투영
        out = self.projection(x_cat)  # (batch_size, output_dim)
        return out


class QsingBertModel(nn.Module):
    def __init__(self):
        super(QsingBertModel, self).__init__()
        
        # self.bert_urls = BertModel.from_pretrained('bert-base-uncased')
        self.bert_urls = UrlCnnModel(
            98,
            output_dim=512
        )
        self.bert_html = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        
        self.fc = nn.Linear(512 * 2, 512)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def median_pooling(self, outputs):
        return torch.median(outputs, dim=1).values

    def forward(self, input):
        url_input_ids = input['url_input_ids']
        url_attention_mask = input['url_attention_mask']
        html_input_ids = input['html_input_ids']
        html_attention_mask = input['html_attention_mask']
        
        url_cls_embedding = self.bert_urls(input_ids=url_input_ids, attention_mask=url_attention_mask)
        # url_cls_embedding = url_output.last_hidden_state[:, 0, :]
        # url_cls_embedding = self.median_pooling(url_output.last_hidden_state)
        # print(url_cls_embedding.isnan().any())
        
        html_output = self.bert_html(input_ids=html_input_ids, attention_mask=html_attention_mask)
        # html_cls_embedding = html_output.last_hidden_state[:, 0, :]
        html_cls_embedding = self.median_pooling(html_output.last_hidden_state)
        # print(html_cls_embedding.isnan().any())

        combined = torch.cat((url_cls_embedding, html_cls_embedding), dim=1)

        x = self.fc(combined)
        x = self.gelu(x)

        logits = self.output_layer(x)
        output = self.sigmoid(logits)

        return logits.squeeze(), output.squeeze()
