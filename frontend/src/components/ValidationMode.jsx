import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, Download, AlertCircle, CheckCircle, Target, TrendingUp, RotateCcw, X, Percent, Crosshair, Activity, Zap } from 'lucide-react';
import { validateFile, downloadFile } from '../api';

const ValidationMode = () => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.name.endsWith('.csv') || droppedFile.name.endsWith('.xlsx') || droppedFile.name.endsWith('.xls'))) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Format non supporté. Utilisez CSV ou Excel.');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await validateFile(file);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la validation');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const getMetricColor = (value) => {
    if (value >= 0.9) return 'text-risk-low';
    if (value >= 0.7) return 'text-gold';
    if (value >= 0.5) return 'text-risk-high';
    return 'text-risk-critical';
  };

  const getMetricBg = (value) => {
    if (value >= 0.9) return 'bg-risk-low/10 border-risk-low/30';
    if (value >= 0.7) return 'bg-gold/10 border-gold/30';
    if (value >= 0.5) return 'bg-risk-high/10 border-risk-high/30';
    return 'bg-risk-critical/10 border-risk-critical/30';
  };

  const metrics = result?.metrics ? [
    { key: 'accuracy', label: 'Accuracy', icon: Target, value: result.metrics.accuracy },
    { key: 'precision', label: 'Precision', icon: Crosshair, value: result.metrics.precision },
    { key: 'recall', label: 'Recall', icon: Activity, value: result.metrics.recall },
    { key: 'f1_score', label: 'F1 Score', icon: Zap, value: result.metrics.f1_score },
    { key: 'roc_auc', label: 'ROC AUC', icon: TrendingUp, value: result.metrics.roc_auc },
  ] : [];

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <h2 className="text-3xl font-bold text-pearl mb-3">
          Mode <span className="text-gradient-gold">Validation</span>
        </h2>
        <p className="text-steel max-w-2xl mx-auto">
          Évaluez les performances du modèle avec un fichier contenant les vraies valeurs (colonne Exited)
        </p>
      </motion.div>

      <AnimatePresence mode="wait">
        {!result ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="max-w-2xl mx-auto"
          >
            {/* Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`card relative overflow-hidden transition-all duration-300 ${
                isDragging
                  ? 'border-gold border-2 bg-gold/5'
                  : file
                  ? 'border-risk-low border bg-risk-low/5'
                  : 'border-dashed border-2 border-graphite hover:border-steel'
              }`}
            >
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />

              <div className="flex flex-col items-center py-12">
                <motion.div
                  animate={{ scale: isDragging ? 1.1 : 1 }}
                  className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-6 ${
                    file
                      ? 'bg-risk-low/10 border border-risk-low/30'
                      : 'bg-gradient-to-br from-accent-purple/10 to-gold/10 border border-accent-purple/20'
                  }`}
                >
                  {file ? (
                    <FileSpreadsheet className="w-10 h-10 text-risk-low" />
                  ) : (
                    <Upload className="w-10 h-10 text-accent-purple" />
                  )}
                </motion.div>

                {file ? (
                  <>
                    <h3 className="text-xl font-semibold text-pearl mb-2">Fichier sélectionné</h3>
                    <p className="text-risk-low font-mono">{file.name}</p>
                    <p className="text-steel text-sm mt-1">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </>
                ) : (
                  <>
                    <h3 className="text-xl font-semibold text-pearl mb-2">
                      {isDragging ? 'Déposez le fichier' : 'Fichier de validation'}
                    </h3>
                    <p className="text-steel">Glissez-déposez ou cliquez pour sélectionner</p>
                  </>
                )}
              </div>

              {file && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="absolute top-4 right-4 z-20 p-2 rounded-lg bg-obsidian/80 text-steel hover:text-pearl transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-4 rounded-xl bg-risk-critical/10 border border-risk-critical/30 flex items-center gap-3"
              >
                <AlertCircle className="w-5 h-5 text-risk-critical flex-shrink-0" />
                <p className="text-risk-critical">{error}</p>
              </motion.div>
            )}

            {/* Info Box */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="mt-6 p-5 rounded-xl bg-accent-purple/5 border border-accent-purple/20"
            >
              <h4 className="text-pearl font-semibold mb-3 flex items-center gap-2">
                <Target className="w-4 h-4 text-accent-purple" />
                Format attendu
              </h4>
              <p className="text-steel text-sm mb-3">
                Votre fichier doit contenir toutes les colonnes + la colonne <strong className="text-accent-purple">Exited</strong> (0 ou 1) :
              </p>
              <div className="flex flex-wrap gap-2">
                {['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].map((col) => (
                  <span key={col} className="px-2 py-1 rounded bg-graphite/50 text-silver text-xs font-mono">
                    {col}
                  </span>
                ))}
                <span className="px-2 py-1 rounded bg-accent-purple/20 text-accent-purple text-xs font-mono font-bold">
                  Exited
                </span>
              </div>
            </motion.div>

            {/* Submit Button */}
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              onClick={handleSubmit}
              disabled={!file || loading}
              className="btn-primary w-full mt-6 flex items-center justify-center gap-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-midnight/30 border-t-midnight rounded-full animate-spin" />
                  Validation en cours...
                </>
              ) : (
                <>
                  <Target className="w-5 h-5" />
                  Valider le modèle
                </>
              )}
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {/* Metrics Cards */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
              {metrics.map((metric, index) => (
                <motion.div
                  key={metric.key}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`card border ${getMetricBg(metric.value)}`}
                >
                  <div className="flex flex-col items-center text-center">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-3 ${getMetricBg(metric.value)}`}>
                      <metric.icon className={`w-6 h-6 ${getMetricColor(metric.value)}`} />
                    </div>
                    <p className="text-steel text-sm mb-1">{metric.label}</p>
                    <p className={`text-3xl font-bold font-mono ${getMetricColor(metric.value)}`}>
                      {(metric.value * 100).toFixed(1)}%
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Summary */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="card mb-8"
            >
              <h3 className="text-lg font-semibold text-pearl mb-6">Résumé de la validation</h3>
              <div className="grid sm:grid-cols-4 gap-6">
                <div className="text-center p-4 rounded-xl bg-obsidian/50">
                  <p className="text-steel text-sm mb-2">Total</p>
                  <p className="text-3xl font-bold text-pearl">{result.summary.total}</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-risk-low/10">
                  <p className="text-steel text-sm mb-2">Corrects</p>
                  <p className="text-3xl font-bold text-risk-low">{result.summary.correct}</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-risk-critical/10">
                  <p className="text-steel text-sm mb-2">Incorrects</p>
                  <p className="text-3xl font-bold text-risk-critical">{result.summary.incorrect}</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-gold/10">
                  <p className="text-steel text-sm mb-2">Accuracy</p>
                  <p className="text-3xl font-bold text-gold">{result.summary.accuracy_pct}</p>
                </div>
              </div>

              {/* Visual accuracy bar */}
              <div className="mt-6">
                <div className="flex justify-between mb-2">
                  <span className="text-silver text-sm">Taux de réussite</span>
                  <span className="text-pearl font-mono">{result.summary.correct} / {result.summary.total}</span>
                </div>
                <div className="h-4 bg-graphite rounded-full overflow-hidden flex">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(result.summary.correct / result.summary.total) * 100}%` }}
                    transition={{ duration: 1, delay: 0.3 }}
                    className="h-full bg-gradient-to-r from-risk-low to-emerald-400"
                  />
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(result.summary.incorrect / result.summary.total) * 100}%` }}
                    transition={{ duration: 1, delay: 0.3 }}
                    className="h-full bg-gradient-to-r from-risk-critical to-red-400"
                  />
                </div>
                <div className="flex justify-between mt-2 text-xs">
                  <span className="text-risk-low flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Corrects
                  </span>
                  <span className="text-risk-critical flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" /> Incorrects
                  </span>
                </div>
              </div>
            </motion.div>

            {/* Predictions Table (if inline mode) */}
            {result.mode === 'inline' && result.predictions && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="card mb-8"
              >
                <h3 className="text-lg font-semibold text-pearl mb-6">Détail des prédictions</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-graphite">
                        <th className="text-left py-3 px-4 text-silver font-medium">ID</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Réel</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Prédit</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Probabilité</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Risque</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Résultat</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.predictions.map((pred, index) => (
                        <tr key={index} className="border-b border-graphite/50 hover:bg-obsidian/30">
                          <td className="py-3 px-4 text-pearl font-mono">{pred.customer_id}</td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              pred.y_true === 1
                                ? 'bg-risk-critical/10 text-risk-critical'
                                : 'bg-risk-low/10 text-risk-low'
                            }`}>
                              {pred.y_true === 1 ? 'Parti' : 'Resté'}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              pred.y_pred === 1
                                ? 'bg-risk-critical/10 text-risk-critical'
                                : 'bg-risk-low/10 text-risk-low'
                            }`}>
                              {pred.y_pred === 1 ? 'Départ' : 'Reste'}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-pearl font-mono">
                            {(pred.probability * 100).toFixed(1)}%
                          </td>
                          <td className="py-3 px-4 text-steel">{pred.risk_level}</td>
                          <td className="py-3 px-4">
                            {pred.correct ? (
                              <CheckCircle className="w-5 h-5 text-risk-low" />
                            ) : (
                              <AlertCircle className="w-5 h-5 text-risk-critical" />
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {/* Download & Reset */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="flex flex-col sm:flex-row gap-4"
            >
              {result.file && (
                <a
                  href={downloadFile(result.file.filename)}
                  download
                  className="btn-primary flex-1 flex items-center justify-center gap-3"
                >
                  <Download className="w-5 h-5" />
                  Télécharger les résultats ({result.file.size_mb} MB)
                </a>
              )}
              <button onClick={handleReset} className="btn-secondary flex items-center justify-center gap-2">
                <RotateCcw className="w-4 h-4" />
                Nouvelle validation
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ValidationMode;
