import { Activity, AlertCircle, FileText, Loader2, Search, TrendingUp, X } from 'lucide-react';
import { useEffect, useState } from 'react';

// Common symptoms for autocomplete
const commonSymptoms = [
  'fever', 'cough', 'fatigue', 'headache', 'sore throat', 'runny nose', 
  'body aches', 'chills', 'nausea', 'vomiting', 'diarrhea', 'abdominal pain',
  'chest pain', 'shortness of breath', 'dizziness', 'rash', 'joint pain',
  'loss of appetite', 'weight loss', 'night sweats', 'confusion', 'weakness'
];

const Prediction = () => {
  const [symptoms, setSymptoms] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedDisease, setSelectedDisease] = useState(null);
  const [treatmentInfo, setTreatmentInfo] = useState(null);
  const [treatmentLoading, setTreatmentLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [apiUrl, setApiUrl] = useState('http://localhost:5000');
  const [comparisonMode, setComparisonMode] = useState(false);

  // Load history from state
  useEffect(() => {
    const saved = [];
  }, []);

  // Autocomplete logic
  const handleSymptomChange = (e) => {
    const value = e.target.value;
    setSymptoms(value);
    
    if (value.length > 1) {
      const lastWord = value.split(',').pop().trim().toLowerCase();
      const matches = commonSymptoms.filter(s => 
        s.toLowerCase().includes(lastWord) && !value.toLowerCase().includes(s)
      );
      setSuggestions(matches.slice(0, 5));
    } else {
      setSuggestions([]);
    }
  };

  const addSymptom = (symptom) => {
    const words = symptoms.split(',').map(s => s.trim()).filter(s => s);
    words.pop(); // Remove incomplete word
    words.push(symptom);
    setSymptoms(words.join(', ') + ', ');
    setSuggestions([]);
  };

  // Predict diseases
  const handlePredict = async () => {
    if (!symptoms.trim()) {
      setError('Please enter at least one symptom');
      return;
    }

    setLoading(true);
    setError('');
    setPredictions(null);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: symptoms.trim() })
      });

      if (!response.ok) throw new Error('Prediction failed');
      
      const data = await response.json();
      setPredictions(data);
      
    } catch (err) {
      setError('Failed to get predictions. Please check your API connection.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Get treatment info
  const handleGetTreatment = async (disease) => {
    setSelectedDisease(disease);
    setTreatmentLoading(true);
    setShowModal(true);
    setTreatmentInfo(null);

    try {
      const response = await fetch(`${apiUrl}/treatment?disease=${disease}`, {
        method: 'get',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) throw new Error('Treatment fetch failed');
      
      const data = await response.json();
      // console.log(data);
      setTreatmentInfo(data.treatment);
      
    } catch (err) {
      setTreatmentInfo({ error: 'Failed to fetch treatment information' });
      console.error(err);
    } finally {
      setTreatmentLoading(false);
    }
  };

  // Prepare chart data
  const getChartData = () => {
    if (!predictions) return [];
    
    const cosine = predictions.cosine?.slice(0, 10) || [];
    const tfidf = predictions.tfidf?.slice(0, 10) || [];
    
    const dataMap = new Map();
    
    cosine.forEach(item => {
      dataMap.set(item.disease, { 
        disease: item.disease, 
        cosine: parseFloat(item.score.toFixed(3)),
        tfidf: 0 
      });
    });
    
    tfidf.forEach(item => {
      if (dataMap.has(item.disease)) {
        dataMap.get(item.disease).tfidf = parseFloat(item.score.toFixed(3));
      } else {
        dataMap.set(item.disease, { 
          disease: item.disease, 
          cosine: 0,
          tfidf: parseFloat(item.score.toFixed(3))
        });
      }
    });
    
    return Array.from(dataMap.values());
  };

  // Export as text
  const handleExport = () => {
    if (!predictions) return;
    
    let text = `Disease Prediction Report\n`;
    text += `Symptoms: ${symptoms}\n`;
    text += `Date: ${new Date().toLocaleString()}\n\n`;
    text += `Top Predictions (Cosine Similarity):\n`;
    predictions.cosine?.slice(0, 10).forEach((p, i) => {
      text += `${i + 1}. ${p.disease} - Score: ${p.score.toFixed(3)}\n`;
    });
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'disease-prediction.txt';
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Disease Predictor</h1>
                <p className="text-sm text-gray-500">AI-powered symptom analysis</p>
              </div>
            </div>
            
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 sticky top-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Search className="w-5 h-5 text-blue-600" />
                Enter Symptoms
              </h2>
              
              <div className="space-y-4">
                <div className="relative">
                  <textarea
                    value={symptoms}
                    onChange={handleSymptomChange}
                    placeholder="Type symptoms separated by commas (e.g., fever, cough, fatigue)"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    rows="4"
                  />
                  
                  {suggestions.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg">
                      {suggestions.map((symptom, idx) => (
                        <div
                          key={idx}
                          onClick={() => addSymptom(symptom)}
                          className="px-4 py-2 hover:bg-blue-50 cursor-pointer text-sm"
                        >
                          {symptom}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={handlePredict}
                  disabled={loading || !symptoms.trim()}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="w-5 h-5" />
                      Predict Diseases
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2 space-y-6">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-900">Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            )}

            {predictions && (
              <>
                  {/* Table View */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Cosine Predictions */}
                    <div className="bg-white rounded-lg shadow-md overflow-hidden">
                      <div className="bg-blue-600 px-4 py-3">
                        <h3 className="font-semibold text-white">Cosine Similarity(Score : 0 - 1)</h3>
                      </div>
                      <div className="divide-y divide-gray-200">
                        {predictions.cosine?.slice(0, 10).map((pred, idx) => (
                          <div
                            key={idx}
                            onClick={() => handleGetTreatment(pred.disease)}
                            className={`p-4 hover:bg-blue-50 cursor-pointer transition ${
                              idx === 0 ? 'bg-blue-50 border-l-4 border-blue-600' : ''
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <div className="flex-1">
                                <div className="font-medium text-gray-900">{pred.disease}</div>
                                <div className="text-sm text-gray-500">
                                  Rank #{idx + 1}
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-lg font-bold text-blue-600">
                                  {(pred.score * 100).toFixed(1)}%
                                </div>
                              </div>
                            </div>
                            <div className="mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">
                              <div
                                className="bg-blue-600 h-full rounded-full transition-all"
                                style={{ width: `${pred.score * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* TF-IDF Predictions */}
                    <div className="bg-white rounded-lg shadow-md overflow-hidden">
                      <div className="bg-purple-600 px-4 py-3">
                        <h3 className="font-semibold text-white">TF-IDF Method(Score : 0 - ∞)</h3>
                      </div>
                      <div className="divide-y divide-gray-200">
                        {predictions.tfidf?.slice(0, 10).map((pred, idx) => (
                          <div
                            key={idx}
                            onClick={() => handleGetTreatment(pred.disease)}
                            className={`p-4 hover:bg-purple-50 cursor-pointer transition ${
                              idx === 0 ? 'bg-purple-50 border-l-4 border-purple-600' : ''
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <div className="flex-1">
                                <div className="font-medium text-gray-900">{pred.disease}</div>
                                <div className="text-sm text-gray-500">
                                  Rank #{idx + 1}
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-lg font-bold text-purple-600">
                                  {((pred.score).toFixed(1))}
                                </div>
                              </div>
                            </div>
                            <div className="mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">
                              <div
                                className="bg-purple-600 h-full rounded-full transition-all"
                                style={{ width: `${(((pred.score / predictions.tfidf[0].score) * 100).toFixed(1))}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
              </>
            )}

            {!predictions && !loading && !error && (
              <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Predictions Yet</h3>
                <p className="text-gray-500">Enter your symptoms to get started</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Treatment Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 flex justify-between items-center">
              <div className="flex items-center gap-3">
                <FileText className="w-6 h-6 text-white" />
                <h3 className="text-xl font-bold text-white">{selectedDisease}</h3>
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="text-white hover:text-gray-200"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)]">
              {treatmentLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                </div>
              ) : treatmentInfo?.error ? (
                <div className="text-red-600 text-center py-8">
                  {treatmentInfo.error}
                </div>
              ) : treatmentInfo ? (
                <div className="space-y-4">
                  <div className="prose max-w-none">
                    <p className="text-gray-700 leading-relaxed">
                      {treatmentInfo.summary || treatmentInfo.treatment || 'No treatment information available.'}
                    </p>
                  </div>
                  
                  {treatmentInfo.url && (
                    <div className="pt-4 border-t">
                      <a
                        href={treatmentInfo.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
                      >
                        Read full article on Wikipedia →
                      </a>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Prediction;