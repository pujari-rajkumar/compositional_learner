import torch

__author__ = "Rajkumar Pujari"

class Node():

    def __init__(self, documents, tokenizer, model, meta_data={}):
        self.document_ids = {}    #Document indices to doc-path dict
        self.doc_idx = 0          #Running next document index
        self.doc_types = {}       #Doc-id to doc-type (description, generated, etc.,)
        self.meta_data = {}       #Meta-data associated with the node
        self.summary_doc_tensor = None      #PLM tensors of all sentences of all documents n_docs * n_sents * h_dim
        self._compute_doc_embeddings(documents, tokenizer, model)


    def _compute_doc_embeddings(self, documents, tokenizer, model):
        edoc_tensors = []
        for doc_path in documents:
            self.document_ids[self.doc_idx] = doc_path
            doc_type, doc_text = documents[doc_path]
            self.doc_types[self.doc_idx] = doc_type
            self.doc_idx += 1
            doc_tensors = []
            for para in doc_text:
                p_tensors = []
                for sent in para:
                    s_toks = tokenizer.encode(sent, add_special_tokens=True)
                    s_tensor = model(torch.tesnor(sent[:512].view(1, -1)))[0]
                    s_tensor = s_tensor.mean(s_tensor, dim=1)
                    p_tensors.append(s_tensor)
                p_tensor = torch.cat(p_tensors, dim=0)
                doc_tensors.append(p_tensor)
            doc_tensor = torch.cat(doc_tensors, dim=0)
            doc_tensor = doc_tensor.unsqueeze(0)
            edoc_tensors.append(doc_tensor)

        edoc_tensor = torch.cat(edoc_tensors, dim=0) 
        self.summary_doc_tensor = edoc_tensor


    def add_document(self, doc_path, doc_text, doc_type, tokenizer, model):
        self.document_ids[doc_path] = self.doc_idx
        self.doc_types[self.doc_idx] = doc_type
        self.doc_idx += 1
        s_tensors = []
        for para in doc_text:
            for sent in para:
                s_toks = tokenizer.encode(sent, add_special_tokens=True)
                s_tensor = model(torch.tesnor(sent[:512].view(1, -1)))[0]
                s_tensor = s_tensor.mean(s_tensor, dim=1)
                s_tensors.append(s_tensor)

        doc_tensor = torch.cat(s_tensors, dim=0)
        self.summary_doc_tensor = torch.cat([self.summary_doc_tensor, doc_tensor], dim=0)






            
