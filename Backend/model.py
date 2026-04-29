import torch
import torch.nn as nn
import torchvision.models as models

class CNNRNN(nn.Module):
    def __init__(self, num_classes=20, embed_dim=64, lstm_hidden=512, dropout=0.5):
        super().__init__()
        self.C = num_classes
        self.START = num_classes
        self.END = num_classes + 1

        self.label_embed = nn.Embedding(num_classes + 2, embed_dim)

        # CNN (VGG16)
        vgg = models.vgg16(pretrained=True)
        self.cnn_features = vgg.features
        self.cnn_avgpool = vgg.avgpool
        self.cnn_fc = nn.Sequential(*list(vgg.classifier.children())[:-2])
        
        self.img_dim = 4096

        # LSTM
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        # Projections
        self.proj_rnn = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden, embed_dim))
        self.proj_img = nn.Sequential(nn.Dropout(dropout), nn.Linear(4096, embed_dim))

    def extract_img(self, x):
        with torch.no_grad():
            f = self.cnn_features(x)
        f = self.cnn_avgpool(f)
        return self.cnn_fc(f.flatten(1))

    def get_W(self):
        W = self.label_embed.weight.clone()
        W[self.START] = -1e9
        return W

    @torch.no_grad()
    def predict_beam(self, images, beam_size=2):
        B = images.size(0)
        device = images.device
        img_emb = self.proj_img(self.extract_img(images))
        results = []

        for b in range(B):
            ie = img_emb[b:b+1]
            tok = torch.tensor([[self.START]], device=device)
            emb = self.label_embed(tok)
            out, hidden = self.lstm(emb)
            x = torch.relu(self.proj_rnn(out.squeeze(1)) + ie)
            
            scores = x @ self.get_W().t()
            log_probs = torch.log_softmax(scores, dim=-1)[0]
            top_lp, top_idx = log_probs.topk(beam_size)

            intermediate_paths = [(top_lp[i].item(), [top_idx[i].item()], hidden) for i in range(beam_size)]
            candidate_paths = []

            while intermediate_paths:
                new_beams = []
                for score, seq, hid in intermediate_paths:
                    if seq[-1] == self.END or len(seq) > self.C:
                        candidate_paths.append((score, seq))
                        continue

                    last = torch.tensor([[seq[-1]]], device=device)
                    emb = self.label_embed(last)
                    out, new_hid = self.lstm(emb, hid)
                    x = torch.relu(self.proj_rnn(out.squeeze(1)) + ie)
                    
                    scores = x @ self.get_W().t()
                    log_probs = torch.log_softmax(scores, dim=-1)[0]
                    top_lp, top_idx = log_probs.topk(beam_size)

                    for i in range(beam_size):
                        lbl = top_idx[i].item()
                        if lbl not in seq:
                            new_beams.append((score + top_lp[i].item(), seq + [lbl], new_hid))

                if not new_beams: break
                new_beams.sort(key=lambda x: x[0], reverse=True)
                intermediate_paths = new_beams[:beam_size]

                if candidate_paths:
                    min_candidate_prob = min([p[0] for p in candidate_paths])
                    if intermediate_paths[0][0] < min_candidate_prob: break

            candidate_paths.extend([(p[0], p[1]) for p in intermediate_paths if p[1][-1] != self.END])
            candidate_paths.sort(key=lambda x: x[0], reverse=True)
            best_seq = candidate_paths[0][1] if candidate_paths else []
            results.append([l for l in best_seq if l < self.C])

        return results