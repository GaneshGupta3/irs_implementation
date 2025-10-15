"""
Disease Information Retrieval System - Complete Backend
Combines ML-based disease prediction with TF-IDF information retrieval
Based on research from IIIT Delhi projects
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import pickle

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

class DiseaseRetrievalSystem:
    """
    Main system class combining symptom-based prediction and 
    document-based information retrieval
    """
    
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.disease_docs = {}
        self.symptom_vectorizer = TfidfVectorizer()
        
    def preprocess_text(self, text):
        """Preprocess text: lowercase, remove stopwords, stem"""
        words = text.lower().split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def load_symptom_dataset(self, filepath):
        """
        Load disease-symptom dataset
        Expected format: CSV with columns for disease and symptoms
        """
        df = pd.read_csv(filepath)
        return df
    
    def train_ml_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple ML models and evaluate performance
        Returns best performing model
        """
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        print(f"\n✓ Best Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
        
        self.models = results
        return results[best_model_name]['model'], best_model_name
    
    def prepare_symptom_data(self, df):
        """
        Prepare symptom data for training
        Assumes df has 'disease' column and symptom columns
        """
        # Get all symptom columns (assuming they start with 'Symptom_')
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_') or col != 'Disease']
        
        # Create feature matrix
        X = df[symptom_cols].fillna('')
        
        # Combine symptoms into single text
        X_text = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        # Preprocess
        X_processed = X_text.apply(self.preprocess_text)
        
        # Vectorize
        X_vectorized = self.symptom_vectorizer.fit_transform(X_processed)
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['Disease'])
        
        return X_vectorized, y
    
    def load_disease_documents(self, folder_path):
        """
        Load disease information documents for TF-IDF retrieval
        Expected format: Text files with disease information
        """
        docs = []
        filenames = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    doc = file.read()
                    docs.append(doc)
                    filenames.append(filename)
                    
                    # Parse structured information
                    disease_info = self.parse_disease_document(doc)
                    self.disease_docs[filename.replace('.txt', '')] = disease_info
        
        # Preprocess documents
        preprocessed_docs = [self.preprocess_text(doc) for doc in docs]
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)
        self.doc_filenames = filenames
        self.raw_docs = docs
        
        print(f"Loaded {len(docs)} disease documents")
    
    def parse_disease_document(self, doc):
        """Parse structured disease information from document"""
        info = {
            'name': '',
            'prevalence': '',
            'risk_factors': '',
            'symptoms': '',
            'treatments': '',
            'preventive_measures': ''
        }
        
        for line in doc.split('\n'):
            if line.startswith('Disease Name:'):
                info['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Prevalence:'):
                info['prevalence'] = line.split(':', 1)[1].strip()
            elif line.startswith('Risk Factors:'):
                info['risk_factors'] = line.split(':', 1)[1].strip()
            elif line.startswith('Symptoms:'):
                info['symptoms'] = line.split(':', 1)[1].strip()
            elif line.startswith('Treatments:'):
                info['treatments'] = line.split(':', 1)[1].strip()
            elif line.startswith('Preventive Measures:'):
                info['preventive_measures'] = line.split(':', 1)[1].strip()
        
        return info
    
    def predict_disease(self, symptoms, model_name='best', top_k=5):
        """
        Predict disease based on symptoms using trained ML model
        """
        # Preprocess input
        processed_symptoms = self.preprocess_text(symptoms)
        symptom_vector = self.symptom_vectorizer.transform([processed_symptoms])
        
        # Get model
        if model_name == 'best':
            model = max(self.models.values(), key=lambda x: x['accuracy'])['model']
        else:
            model = self.models[model_name]['model']
        
        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(symptom_vector)[0]
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            predictions = []
            for idx in top_indices:
                disease = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx]
                predictions.append({
                    'disease': disease,
                    'confidence': confidence
                })
        else:
            # For models without probability
            prediction = model.predict(symptom_vector)[0]
            disease = self.label_encoder.inverse_transform([prediction])[0]
            predictions = [{'disease': disease, 'confidence': 1.0}]
        
        return predictions
    
    def retrieve_disease_info(self, query, top_k=3):
        """
        Retrieve disease information using TF-IDF similarity
        """
        # Preprocess query
        processed_query = self.preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        cosine_similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        
        # Get top results
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:
                disease_name = self.doc_filenames[idx].replace('.txt', '')
                results.append({
                    'disease': disease_name,
                    'similarity': cosine_similarities[idx],
                    'info': self.disease_docs.get(disease_name, {}),
                    'raw_doc': self.raw_docs[idx]
                })
        
        return results
    
    def hybrid_search(self, symptoms, top_k=5):
        """
        Combine ML prediction and TF-IDF retrieval for better results
        """
        # Get ML predictions
        ml_predictions = self.predict_disease(symptoms, top_k=top_k)
        
        # Get TF-IDF retrieval results
        retrieval_results = self.retrieve_disease_info(symptoms, top_k=top_k)
        
        # Combine and rank results
        combined = {}
        
        for pred in ml_predictions:
            disease = pred['disease']
            combined[disease] = {
                'disease': disease,
                'ml_confidence': pred['confidence'],
                'retrieval_score': 0,
                'combined_score': pred['confidence'] * 0.6  # Weight ML more
            }
        
        for result in retrieval_results:
            disease = result['disease']
            if disease in combined:
                combined[disease]['retrieval_score'] = result['similarity']
                combined[disease]['combined_score'] += result['similarity'] * 0.4
            else:
                combined[disease] = {
                    'disease': disease,
                    'ml_confidence': 0,
                    'retrieval_score': result['similarity'],
                    'combined_score': result['similarity'] * 0.4
                }
            combined[disease]['info'] = result['info']
        
        # Sort by combined score
        sorted_results = sorted(combined.values(), 
                              key=lambda x: x['combined_score'], 
                              reverse=True)
        
        return sorted_results[:top_k]
    
    def save_model(self, filepath='disease_model.pkl'):
        """Save trained model and vectorizers"""
        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'symptom_vectorizer': self.symptom_vectorizer,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'disease_docs': self.disease_docs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='disease_model.pkl'):
        """Load trained model and vectorizers"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.label_encoder = model_data['label_encoder']
        self.symptom_vectorizer = model_data['symptom_vectorizer']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.disease_docs = model_data['disease_docs']
        print(f"Model loaded from {filepath}")


# Example usage and demo
def main():
    """
    Main function demonstrating system usage
    """
    print("=" * 60)
    print("Disease Information Retrieval System")
    print("=" * 60)
    
    # Initialize system
    system = DiseaseRetrievalSystem()
    
    # Example 1: Training on symptom dataset
    print("\n[1] Training ML Models on Symptom Dataset")
    print("-" * 60)
    
    # Load your dataset (you'll need to provide the actual file)
    # df = system.load_symptom_dataset('disease_symptom_dataset.csv')
    # X, y = system.prepare_symptom_data(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # best_model, model_name = system.train_ml_models(X_train, X_test, y_train, y_test)
    
    # Example 2: Load disease information documents
    print("\n[2] Loading Disease Information Documents")
    print("-" * 60)
    
    # system.load_disease_documents('Disease_Documents')
    
    # Example 3: Interactive symptom checker
    print("\n[3] Interactive Symptom Checker")
    print("-" * 60)
    
    # Simulated predictions (replace with actual after training)
    sample_symptoms = "fever, cough, body aches, headache, fatigue"
    print(f"\nInput Symptoms: {sample_symptoms}")
    print("\nAnalyzing symptoms...")
    
    # This would use actual trained model:
    # predictions = system.predict_disease(sample_symptoms, top_k=5)
    # for i, pred in enumerate(predictions, 1):
    #     print(f"{i}. {pred['disease']}: {pred['confidence']*100:.2f}% confidence")
    
    # Example 4: Document retrieval
    print("\n[4] Disease Information Retrieval")
    print("-" * 60)
    
    # query = "What are the symptoms and treatments for influenza?"
    # results = system.retrieve_disease_info(query, top_k=3)
    # for result in results:
    #     print(f"\nDisease: {result['info']['name']}")
    #     print(f"Similarity Score: {result['similarity']:.4f}")
    #     print(f"Symptoms: {result['info']['symptoms']}")
    
    # Example 5: Hybrid search
    print("\n[5] Hybrid Search (ML + TF-IDF)")
    print("-" * 60)
    
    # hybrid_results = system.hybrid_search(sample_symptoms, top_k=5)
    # for i, result in enumerate(hybrid_results, 1):
    #     print(f"\n{i}. {result['disease']}")
    #     print(f"   Combined Score: {result['combined_score']:.4f}")
    #     print(f"   ML Confidence: {result['ml_confidence']:.4f}")
    #     print(f"   Retrieval Score: {result['retrieval_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("System Demo Complete!")
    print("=" * 60)


def create_sample_disease_document():
    """
    Create sample disease document format
    This shows the expected format for disease information files
    """
    sample_doc = """Disease Name: Influenza (Flu)
Prevalence: Seasonal, affects 5-20% of population annually worldwide
Risk Factors: Age (young children and elderly), chronic medical conditions, pregnancy, weakened immune system, close contact with infected persons
Symptoms: High fever (100-104°F), severe body aches, fatigue, dry cough, headache, chills, sore throat, nasal congestion, occasional nausea and vomiting
Treatments: Antiviral medications (oseltamivir, zanamivir), rest, plenty of fluids, over-the-counter pain relievers (acetaminophen, ibuprofen), cough suppressants
Preventive Measures: Annual flu vaccination, frequent handwashing, avoid touching face, stay away from sick individuals, maintain good health practices, cover coughs and sneezes
"""
    return sample_doc


def create_sample_dataset():
    """
    Create sample symptom-disease dataset
    Shows the expected CSV format
    """
    data = {
        'Disease': ['Influenza', 'Influenza', 'Common Cold', 'Common Cold', 'Migraine', 'Migraine'],
        'Symptom_1': ['fever', 'fever', 'runny nose', 'sneezing', 'severe headache', 'headache'],
        'Symptom_2': ['body aches', 'cough', 'sore throat', 'cough', 'nausea', 'sensitivity to light'],
        'Symptom_3': ['fatigue', 'headache', 'mild fever', 'congestion', 'visual disturbances', 'nausea'],
        'Symptom_4': ['cough', 'chills', 'sneezing', 'sore throat', '', ''],
        'Symptom_5': ['headache', 'sore throat', '', '', '', '']
    }
    
    df = pd.DataFrame(data)
    return df


# Advanced Features
class AdvancedFeatures:
    """
    Additional features for enhanced system performance
    Based on the advanced techniques from Shreeram's project
    """
    
    @staticmethod
    def synonym_expansion(symptom, synonym_dict):
        """
        Expand symptoms to include synonyms for better matching
        """
        synonyms = synonym_dict.get(symptom.lower(), [symptom])
        return synonyms
    
    @staticmethod
    def calculate_jaccard_similarity(set1, set2):
        """
        Calculate Jaccard similarity between two symptom sets
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def get_related_symptoms(primary_symptom, symptom_network):
        """
        Get commonly co-occurring symptoms
        """
        return symptom_network.get(primary_symptom, [])
    
    @staticmethod
    def iterative_symptom_suggestion(current_symptoms, all_diseases):
        """
        Suggest additional symptoms based on common co-occurrence
        """
        suggestions = set()
        for disease, symptoms in all_diseases.items():
            overlap = set(current_symptoms).intersection(set(symptoms))
            if len(overlap) > 0:
                suggestions.update(set(symptoms) - set(current_symptoms))
        return list(suggestions)[:10]  # Top 10 suggestions


# Symptom synonym dictionary (expand as needed)
SYMPTOM_SYNONYMS = {
    'fever': ['high temperature', 'pyrexia', 'elevated temperature'],
    'headache': ['head pain', 'cephalalgia', 'migraine'],
    'cough': ['coughing', 'tussis', 'hacking'],
    'fatigue': ['tiredness', 'exhaustion', 'weakness', 'lethargy'],
    'nausea': ['queasiness', 'sick feeling', 'upset stomach'],
    'pain': ['ache', 'discomfort', 'soreness'],
    'rash': ['skin eruption', 'skin redness', 'hives'],
    'shortness of breath': ['dyspnea', 'breathing difficulty', 'breathlessness'],
    'dizziness': ['vertigo', 'lightheadedness', 'unsteadiness'],
    'vomiting': ['throwing up', 'emesis', 'being sick']
}


# Evaluation metrics
def evaluate_system(y_true, y_pred, disease_names):
    """
    Comprehensive evaluation of the system
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("SYSTEM EVALUATION METRICS")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=disease_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=disease_names, output_dict=True)
    }


if __name__ == "__main__":
    main()
    
    # Display sample formats
    print("\n\n" + "="*60)
    print("SAMPLE DATA FORMATS")
    print("="*60)
    
    print("\n[Sample Disease Document Format]")
    print("-"*60)
    print(create_sample_disease_document())
    
    print("\n[Sample Dataset Format]")
    print("-"*60)
    print(create_sample_dataset())
    
    print("\n[Symptom Synonyms Available]")
    print("-"*60)
    for symptom, synonyms in list(SYMPTOM_SYNONYMS.items())[:5]:
        print(f"{symptom}: {', '.join(synonyms)}")
    
    print("\n" + "="*60)
    print("Ready to implement! Follow the steps below:")
    print("="*60)
    print("\n1. Prepare your disease-symptom dataset (CSV format)")
    print("2. Create disease information documents (TXT files)")
    print("3. Train the models using the DiseaseRetrievalSystem class")
    print("4. Test with sample symptoms")
    print("5. Deploy as web service or CLI application")
    print("\n" + "="*60)