from typing import List, Dict, Any
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os
from datetime import datetime

class RAGEvaluator:
    def __init__(self, eval_data_path: str = "./data/evaluation/qa_pairs.jsonl"):
        self.eval_data_path = eval_data_path
        os.makedirs(os.path.dirname(eval_data_path), exist_ok=True)
    
    def evaluate_retrieval(self, retrieved_docs: List[Document], 
                          expected_doc_ids: List[str], query: str = "") -> Dict[str, float]:
        """Evaluate retrieval performance"""
        retrieved_ids = [doc.metadata.get('doc_id', str(i)) for i, doc in enumerate(retrieved_docs)]
        
        # Binary relevance (1 if expected doc is in retrieved, 0 otherwise)
        y_true = [1 if doc_id in retrieved_ids else 0 for doc_id in expected_doc_ids]
        y_pred = [1] * len(retrieved_ids)  # All retrieved docs are considered relevant for evaluation
        
        if not y_true:  # No expected documents
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'retrieved_count': len(retrieved_docs),
                'expected_count': 0
            }
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'retrieved_count': len(retrieved_docs),
            'expected_count': len(expected_doc_ids),
            'query': query
        }
    
    def evaluate_answer_quality(self, generated_answer: str, reference_answer: str, 
                               question: str = "") -> Dict[str, float]:
        """Simple answer quality evaluation using text similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer().fit_transform([generated_answer, reference_answer])
        vectors = vectorizer.toarray()
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        
        return {
            'answer_similarity': round(similarity, 3),
            'generated_length': len(generated_answer),
            'reference_length': len(reference_answer),
            'question': question
        }
    
    def add_evaluation_pair(self, question: str, expected_doc_ids: List[str], 
                           reference_answer: str = ""):
        """Add a new QA pair to evaluation dataset"""
        eval_data = {
            'question': question,
            'expected_doc_ids': expected_doc_ids,
            'reference_answer': reference_answer,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.eval_data_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(eval_data) + '\n')
    
    def load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load all evaluation data"""
        if not os.path.exists(self.eval_data_path):
            return []
        
        eval_data = []
        with open(self.eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line.strip()))
        
        return eval_data
    
    def run_comprehensive_evaluation(self, retriever, generator, num_samples: int = 5):
        """Run comprehensive evaluation on stored QA pairs"""
        eval_data = self.load_evaluation_data()[:num_samples]
        results = []
        
        for i, data in enumerate(eval_data):
            print(f"🧪 Evaluating sample {i+1}/{len(eval_data)}: {data['question'][:50]}...")
            
            # Test retrieval
            retrieved_docs = retriever.retrieve(data['question'])
            retrieval_metrics = self.evaluate_retrieval(
                retrieved_docs, data['expected_doc_ids'], data['question']
            )
            
            # Test generation if reference answer exists
            generation_metrics = {}
            if 'reference_answer' in data and data['reference_answer']:
                generated_answer = generator.generate_answer(data['question'], retrieved_docs)
                generation_metrics = self.evaluate_answer_quality(
                    generated_answer, data['reference_answer'], data['question']
                )
            
            results.append({
                'retrieval_metrics': retrieval_metrics,
                'generation_metrics': generation_metrics,
                'sample_data': data
            })
        
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        if not results:
            return {}
        
        retrieval_precisions = [r['retrieval_metrics']['precision'] for r in results]
        retrieval_recalls = [r['retrieval_metrics']['recall'] for r in results]
        retrieval_f1s = [r['retrieval_metrics']['f1_score'] for r in results]
        
        aggregated = {
            'retrieval': {
                'mean_precision': round(np.mean(retrieval_precisions), 3),
                'mean_recall': round(np.mean(retrieval_recalls), 3),
                'mean_f1': round(np.mean(retrieval_f1s), 3),
                'std_precision': round(np.std(retrieval_precisions), 3),
                'sample_count': len(results)
            }
        }
        
        # Add generation metrics if available
        if 'generation_metrics' in results[0] and results[0]['generation_metrics']:
            gen_similarities = [r['generation_metrics']['answer_similarity'] for r in results if 'generation_metrics' in r]
            aggregated['generation'] = {
                'mean_similarity': round(np.mean(gen_similarities), 3),
                'std_similarity': round(np.std(gen_similarities), 3)
            }
        
        return aggregated