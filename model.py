from gensim.models.fasttext import FastText
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def ft_train(word_tokenized_corpus):
    ft_model = FastText(word_tokenized_corpus,
                        vector_size = 224,
                        window = 40,
                        min_count = 5,
                        sample = 1e-2,
                        sg = 1,
                        epochs = 100)
    return ft_model

class PatchEmbedding(nn.Module):
	def __init__(self, in_channels, patch_size, emb_size, img_size):
		super().__init__()
		self.patch_size = patch_size
		self.nPatches = (img_size*img_size) // ((patch_size)**2)
		self.arrange = Rearrange('b c (h p1)(w p2) -> b (h w) (p1 p2 c)',p1 = patch_size,p2 = patch_size)
		self.transform = nn.Linear(patch_size * patch_size * in_channels, emb_size)
		self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

	def forward(self, x: Tensor, text_embeds):
		b,c,h,w = x.shape
		x = self.arrange(x)
		x = self.transform(x)

		cls_tokens = repeat(self.cls_token,'() n e -> b n e', b=b) #repeat the cls tokens for all patch set in

		x = torch.cat([x, text_embeds], dim = 1)
		x = torch.cat([cls_tokens,x],dim=1)

		return x

class multiHeadAttention(nn.Module):
	def __init__(self, emb_size, heads, dropout):
		super().__init__()
		self.heads = heads
		self.emb_size = emb_size
		self.query = nn.Linear(emb_size,emb_size)
		self.key = nn.Linear(emb_size,emb_size)
		self.value = nn.Linear(emb_size,emb_size)
		self.drop_out = nn.Dropout(dropout)
		self.projection = nn.Linear(emb_size,emb_size)

	def forward(self,x):
		#splitting the single input int number of heads
		queries = rearrange(self.query(x),"b n (h d) -> b h n d", h = self.heads)
		keys = rearrange(self.key(x),"b n (h d) -> b h n d", h = self.heads)
		values = rearrange(self.value(x),"b n (h d) -> b h n d", h = self.heads)
		attention_maps = torch.einsum("bhqd, bhkd -> bhqk",queries,keys)
		scaling_value = self.emb_size**(1/2)
		attention_maps = F.softmax(attention_maps,dim=-1)/scaling_value
		attention_maps = self.drop_out(attention_maps) #! might be deleted
		output = torch.einsum("bhal, bhlv -> bhav",attention_maps,values)
		output  = rearrange(output,"b h n d -> b n (h d)")
		output = self.projection(output)
		return output

class residual(nn.Module):
	def __init__(self,fn):
		super().__init__()
		self.fn = fn
	def forward(self,x):
		identity = x
		res = self.fn(x)
		out = res + identity
		return out

class DeepBlock(nn.Sequential):
	def __init__(self, emb_size:int=224, drop_out:float=0.0):#64
		super().__init__(
        		residual(
            			nn.Sequential(
                			nn.LayerNorm(emb_size),
                			multiHeadAttention(emb_size,heads=2,dropout=0),
                			nn.LayerNorm(emb_size)
            			)
        		)
    		)

class Transformation(nn.Sequential):
	def __init__(self, emb_size:int=224, n_classes:int=2):
		super().__init__(
			Reduce('b n e -> b e', reduction='mean'),
			nn.LayerNorm(emb_size),
			nn.Linear(emb_size, emb_size)) # no use of earlier n_classes here since we only wish to have the embeddings

class Model(nn.Module):
	def __init__(self, emb_size:int=224, drop_out:float=0.0, n_classes:int=2, in_channels:int=1, patch_size:int=16, image_size:int=256):
		super().__init__()
		self.PatchEmbedding = PatchEmbedding(in_channels,patch_size,emb_size,image_size)
		self.DeepBlock = DeepBlock(emb_size = emb_size)#Transformer()
		self.Transformation = Transformation(emb_size=emb_size, n_classes=n_classes)
	def forward(self, x, text_embeds):
		patchEmbeddings = self.PatchEmbedding(x, text_embeds)
		DeepBlockOp = self.DeepBlock(patchEmbeddings)
		transformationOutput = self.Transformation(DeepBlockOp)
		return transformationOutput

class attention_model(nn.Module):
  def __init__(self, encoder=None, emb_size:int=224):
    super(attention_model, self).__init__()
	
    if encoder is None:
        encoder = Model(emb_size=emb_size)
    self.encoder = encoder

  def forward(self, images_x, text_embeds):
    combined_features = self.encoder(images_x, text_embeds)
    return combined_features

class classifier(nn.Module):
  def __init__(self, emb_size:int=224, num_classes:int=2, hidden_size:int=128):
        super(classifier, self).__init__()

        self.mid_layer = nn.Linear(emb_size, hidden_size)
        self.real_classifier = nn.Linear(hidden_size, num_classes)
        self.violent_classifier = nn.Linear(hidden_size, num_classes)
        self.sentiment_classifier = nn.Linear(hidden_size, 3)

  def forward(self, combined_features):
        mid_features = self.mid_layer(combined_features)

        real = self.real_classifier(F.relu(mid_features))
        violent = self.violent_classifier(F.relu(mid_features))
        sentiment = self.sentiment_classifier(F.relu(mid_features))

        return real, violent, sentiment
