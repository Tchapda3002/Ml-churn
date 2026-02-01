import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, Download, AlertCircle, CheckCircle, Users, TrendingUp, AlertTriangle, XCircle, RotateCcw, X } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { predictFile, downloadFile } from '../api';

const BatchPrediction = () => {
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
      const response = await predictFile(file);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la prédiction');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const riskColors = {
    Low: '#10b981',
    Medium: '#f59e0b',
    High: '#f97316',
    Critical: '#ef4444',
  };

  const riskIcons = {
    Low: CheckCircle,
    Medium: AlertTriangle,
    High: AlertTriangle,
    Critical: XCircle,
  };

  const pieData = result?.summary?.risk_distribution
    ? Object.entries(result.summary.risk_distribution).map(([name, value]) => ({
        name,
        value,
        color: riskColors[name],
      }))
    : [];

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <h2 className="text-3xl font-bold text-pearl mb-3">
          Prédiction <span className="text-gradient-gold">Batch</span>
        </h2>
        <p className="text-steel max-w-2xl mx-auto">
          Importez un fichier CSV ou Excel pour analyser plusieurs clients simultanément
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
                      : 'bg-gradient-to-br from-gold-dark/10 to-gold/10 border border-gold/20'
                  }`}
                >
                  {file ? (
                    <FileSpreadsheet className="w-10 h-10 text-risk-low" />
                  ) : (
                    <Upload className="w-10 h-10 text-gold" />
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
                      {isDragging ? 'Déposez le fichier' : 'Glissez-déposez votre fichier'}
                    </h3>
                    <p className="text-steel">ou cliquez pour sélectionner</p>
                    <div className="flex items-center gap-4 mt-4">
                      <span className="px-3 py-1 rounded-full bg-obsidian border border-graphite text-sm text-silver">CSV</span>
                      <span className="px-3 py-1 rounded-full bg-obsidian border border-graphite text-sm text-silver">XLSX</span>
                      <span className="px-3 py-1 rounded-full bg-obsidian border border-graphite text-sm text-silver">XLS</span>
                    </div>
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
              className="mt-6 p-5 rounded-xl bg-obsidian/50 border border-graphite/50"
            >
              <h4 className="text-pearl font-semibold mb-3">Format attendu</h4>
              <p className="text-steel text-sm mb-3">
                Votre fichier doit contenir les colonnes suivantes (sans la colonne Exited) :
              </p>
              <div className="flex flex-wrap gap-2">
                {['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].map((col) => (
                  <span key={col} className="px-2 py-1 rounded bg-graphite/50 text-silver text-xs font-mono">
                    {col}
                  </span>
                ))}
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
                  Analyse en cours...
                </>
              ) : (
                <>
                  <TrendingUp className="w-5 h-5" />
                  Lancer l'analyse
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
            {/* Summary Cards */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="card"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-accent-blue/10 border border-accent-blue/30 flex items-center justify-center">
                    <Users className="w-6 h-6 text-accent-blue" />
                  </div>
                  <div>
                    <p className="text-steel text-sm">Total clients</p>
                    <p className="text-2xl font-bold text-pearl">{result.summary.total_customers}</p>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="card"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-risk-critical/10 border border-risk-critical/30 flex items-center justify-center">
                    <AlertTriangle className="w-6 h-6 text-risk-critical" />
                  </div>
                  <div>
                    <p className="text-steel text-sm">Départs prédits</p>
                    <p className="text-2xl font-bold text-risk-critical">{result.summary.churn_predicted}</p>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gold/10 border border-gold/30 flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-gold" />
                  </div>
                  <div>
                    <p className="text-steel text-sm">Taux de churn</p>
                    <p className="text-2xl font-bold text-gold">{(result.summary.churn_rate * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="card"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-accent-purple/10 border border-accent-purple/30 flex items-center justify-center">
                    <FileSpreadsheet className="w-6 h-6 text-accent-purple" />
                  </div>
                  <div>
                    <p className="text-steel text-sm">Probabilité moy.</p>
                    <p className="text-2xl font-bold text-accent-purple">{(result.summary.avg_probability * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Charts */}
            <div className="grid lg:grid-cols-2 gap-6 mb-8">
              {/* Risk Distribution Pie */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
                className="card"
              >
                <h3 className="text-lg font-semibold text-pearl mb-6">Distribution des risques</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={4}
                        dataKey="value"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: '8px',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-wrap justify-center gap-4 mt-4">
                  {pieData.map((entry) => (
                    <div key={entry.name} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                      <span className="text-silver text-sm">{entry.name}: {entry.value}</span>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Risk Bar Chart */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 }}
                className="card"
              >
                <h3 className="text-lg font-semibold text-pearl mb-6">Répartition par niveau de risque</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={pieData} layout="vertical">
                      <XAxis type="number" stroke="#64748b" />
                      <YAxis type="category" dataKey="name" stroke="#64748b" width={80} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: '8px',
                        }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {pieData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            </div>

            {/* Predictions Table (if inline mode) */}
            {result.mode === 'inline' && result.predictions && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="card mb-8"
              >
                <h3 className="text-lg font-semibold text-pearl mb-6">Détail des prédictions</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-graphite">
                        <th className="text-left py-3 px-4 text-silver font-medium">ID</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Prédiction</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Probabilité</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Risque</th>
                        <th className="text-left py-3 px-4 text-silver font-medium">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.predictions.map((pred, index) => {
                        const RiskIcon = riskIcons[pred.risk_level];
                        return (
                          <tr key={index} className="border-b border-graphite/50 hover:bg-obsidian/30">
                            <td className="py-3 px-4 text-pearl font-mono">{pred.customer_id}</td>
                            <td className="py-3 px-4">
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                pred.churn_prediction === 1
                                  ? 'bg-risk-critical/10 text-risk-critical'
                                  : 'bg-risk-low/10 text-risk-low'
                              }`}>
                                {pred.churn_prediction === 1 ? 'Départ' : 'Reste'}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-pearl font-mono">
                              {(pred.churn_probability * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <RiskIcon className="w-4 h-4" style={{ color: riskColors[pred.risk_level] }} />
                                <span style={{ color: riskColors[pred.risk_level] }}>{pred.risk_level}</span>
                              </div>
                            </td>
                            <td className="py-3 px-4 text-steel text-sm max-w-xs truncate">
                              {pred.recommended_action}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {/* Download & Reset */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
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
                Nouvelle analyse
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default BatchPrediction;
