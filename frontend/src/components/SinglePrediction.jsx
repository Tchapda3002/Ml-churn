import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, MapPin, CreditCard, Wallet, Package, Clock, Sparkles, AlertCircle, ArrowRight, RotateCcw, TrendingUp } from 'lucide-react';
import { predictSingle } from '../api';
import RiskGauge from './RiskGauge';

const SinglePrediction = () => {
  const [formData, setFormData] = useState({
    CreditScore: 650,
    Geography: 'France',
    Gender: 'Male',
    Age: 35,
    Tenure: 5,
    Balance: 125000,
    NumOfProducts: 2,
    HasCrCard: 1,
    IsActiveMember: 1,
    EstimatedSalary: 85000,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictSingle(formData);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la prédiction');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.05 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <h2 className="text-3xl font-bold text-pearl mb-3">
          Prédiction <span className="text-gradient-gold">Individuelle</span>
        </h2>
        <p className="text-steel max-w-2xl mx-auto">
          Analysez le risque de départ d'un client en renseignant ses informations
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Form */}
        <motion.form
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          onSubmit={handleSubmit}
          className="card"
        >
          <div className="grid sm:grid-cols-2 gap-5">
            {/* Credit Score */}
            <motion.div variants={itemVariants} className="sm:col-span-2">
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <CreditCard className="w-4 h-4 text-gold" />
                Score de Crédit
              </label>
              <div className="relative">
                <input
                  type="range"
                  min="300"
                  max="850"
                  value={formData.CreditScore}
                  onChange={(e) => handleInputChange('CreditScore', parseInt(e.target.value))}
                  className="w-full h-2 bg-graphite rounded-lg appearance-none cursor-pointer accent-gold"
                />
                <div className="flex justify-between mt-2">
                  <span className="text-xs text-steel">300</span>
                  <span className="text-lg font-mono font-bold text-gold">{formData.CreditScore}</span>
                  <span className="text-xs text-steel">850</span>
                </div>
              </div>
            </motion.div>

            {/* Geography */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <MapPin className="w-4 h-4 text-gold" />
                Pays
              </label>
              <select
                value={formData.Geography}
                onChange={(e) => handleInputChange('Geography', e.target.value)}
                className="input-field cursor-pointer"
              >
                <option value="France">France</option>
                <option value="Germany">Allemagne</option>
                <option value="Spain">Espagne</option>
              </select>
            </motion.div>

            {/* Gender */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <User className="w-4 h-4 text-gold" />
                Genre
              </label>
              <div className="flex gap-3">
                {['Male', 'Female'].map((gender) => (
                  <button
                    key={gender}
                    type="button"
                    onClick={() => handleInputChange('Gender', gender)}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all duration-300 ${
                      formData.Gender === gender
                        ? 'bg-gradient-to-r from-gold-dark/20 to-gold/20 border border-gold/50 text-gold'
                        : 'bg-obsidian/50 border border-graphite/50 text-steel hover:text-pearl'
                    }`}
                  >
                    {gender === 'Male' ? 'Homme' : 'Femme'}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Age */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <User className="w-4 h-4 text-gold" />
                Âge
              </label>
              <input
                type="number"
                min="18"
                max="100"
                value={formData.Age}
                onChange={(e) => handleInputChange('Age', parseInt(e.target.value))}
                className="input-field"
              />
            </motion.div>

            {/* Tenure */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <Clock className="w-4 h-4 text-gold" />
                Ancienneté (ans)
              </label>
              <input
                type="number"
                min="0"
                max="10"
                value={formData.Tenure}
                onChange={(e) => handleInputChange('Tenure', parseInt(e.target.value))}
                className="input-field"
              />
            </motion.div>

            {/* Balance */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <Wallet className="w-4 h-4 text-gold" />
                Solde (€)
              </label>
              <input
                type="number"
                min="0"
                value={formData.Balance}
                onChange={(e) => handleInputChange('Balance', parseFloat(e.target.value))}
                className="input-field"
              />
            </motion.div>

            {/* NumOfProducts */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <Package className="w-4 h-4 text-gold" />
                Produits bancaires
              </label>
              <div className="flex gap-2">
                {[1, 2, 3, 4].map((num) => (
                  <button
                    key={num}
                    type="button"
                    onClick={() => handleInputChange('NumOfProducts', num)}
                    className={`flex-1 py-3 rounded-lg font-mono font-bold transition-all duration-300 ${
                      formData.NumOfProducts === num
                        ? 'bg-gradient-to-r from-gold-dark/20 to-gold/20 border border-gold/50 text-gold'
                        : 'bg-obsidian/50 border border-graphite/50 text-steel hover:text-pearl'
                    }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* HasCrCard */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <CreditCard className="w-4 h-4 text-gold" />
                Carte de crédit
              </label>
              <div className="flex gap-3">
                {[{ value: 1, label: 'Oui' }, { value: 0, label: 'Non' }].map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => handleInputChange('HasCrCard', opt.value)}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all duration-300 ${
                      formData.HasCrCard === opt.value
                        ? 'bg-gradient-to-r from-gold-dark/20 to-gold/20 border border-gold/50 text-gold'
                        : 'bg-obsidian/50 border border-graphite/50 text-steel hover:text-pearl'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* IsActiveMember */}
            <motion.div variants={itemVariants}>
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <Sparkles className="w-4 h-4 text-gold" />
                Membre actif
              </label>
              <div className="flex gap-3">
                {[{ value: 1, label: 'Oui' }, { value: 0, label: 'Non' }].map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => handleInputChange('IsActiveMember', opt.value)}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all duration-300 ${
                      formData.IsActiveMember === opt.value
                        ? 'bg-gradient-to-r from-gold-dark/20 to-gold/20 border border-gold/50 text-gold'
                        : 'bg-obsidian/50 border border-graphite/50 text-steel hover:text-pearl'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* EstimatedSalary */}
            <motion.div variants={itemVariants} className="sm:col-span-2">
              <label className="flex items-center gap-2 text-sm font-medium text-silver mb-2">
                <Wallet className="w-4 h-4 text-gold" />
                Salaire estimé (€)
              </label>
              <input
                type="number"
                min="0"
                value={formData.EstimatedSalary}
                onChange={(e) => handleInputChange('EstimatedSalary', parseFloat(e.target.value))}
                className="input-field"
              />
            </motion.div>
          </div>

          {/* Submit Button */}
          <motion.div variants={itemVariants} className="mt-8">
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-midnight/30 border-t-midnight rounded-full animate-spin" />
                  Analyse en cours...
                </>
              ) : (
                <>
                  Analyser le risque
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </motion.div>
        </motion.form>

        {/* Results */}
        <div className="flex items-stretch">
          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="card flex-1 flex flex-col items-center justify-center text-center"
              >
                <div className="w-16 h-16 rounded-full bg-risk-critical/10 border border-risk-critical/30 flex items-center justify-center mb-4">
                  <AlertCircle className="w-8 h-8 text-risk-critical" />
                </div>
                <h3 className="text-xl font-semibold text-pearl mb-2">Erreur</h3>
                <p className="text-steel mb-6">{error}</p>
                <button onClick={handleReset} className="btn-secondary flex items-center gap-2">
                  <RotateCcw className="w-4 h-4" />
                  Réessayer
                </button>
              </motion.div>
            )}

            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="card flex-1 flex flex-col"
              >
                <div className="text-center mb-6">
                  <h3 className="text-xl font-semibold text-pearl mb-1">Résultat de l'analyse</h3>
                  <p className="text-steel text-sm">Probabilité de départ du client</p>
                </div>

                {/* Risk Gauge */}
                <div className="flex-1 flex items-center justify-center py-4">
                  <RiskGauge probability={result.churn_probability} riskLevel={result.risk_level} />
                </div>

                {/* Prediction */}
                <div className={`text-center py-4 px-6 rounded-xl mb-6 ${
                  result.churn_prediction === 1
                    ? 'bg-risk-critical/10 border border-risk-critical/30'
                    : 'bg-risk-low/10 border border-risk-low/30'
                }`}>
                  <p className={`text-lg font-semibold ${
                    result.churn_prediction === 1 ? 'text-risk-critical' : 'text-risk-low'
                  }`}>
                    {result.churn_prediction === 1 ? '⚠️ Client susceptible de partir' : '✓ Client susceptible de rester'}
                  </p>
                </div>

                {/* Confidence */}
                <div className="bg-obsidian/50 rounded-xl p-4 mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-silver text-sm">Confiance du modèle</span>
                    <span className="text-pearl font-mono font-bold">{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <div className="h-2 bg-graphite rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence * 100}%` }}
                      transition={{ duration: 1, delay: 0.5 }}
                      className="h-full bg-gradient-to-r from-gold-dark to-gold rounded-full"
                    />
                  </div>
                </div>

                {/* Recommended Action */}
                <div className="bg-gradient-to-r from-gold-dark/10 to-gold/10 border border-gold/20 rounded-xl p-5">
                  <h4 className="text-gold font-semibold mb-2 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Action recommandée
                  </h4>
                  <p className="text-pearl">{result.recommended_action}</p>
                </div>

                {/* Reset Button */}
                <button
                  onClick={handleReset}
                  className="btn-secondary mt-6 flex items-center justify-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Nouvelle analyse
                </button>
              </motion.div>
            )}

            {!result && !error && (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="card flex-1 flex flex-col items-center justify-center text-center"
              >
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-gold-dark/10 to-gold/10 border border-gold/20 flex items-center justify-center mb-6 animate-float">
                  <TrendingUp className="w-10 h-10 text-gold" />
                </div>
                <h3 className="text-xl font-semibold text-pearl mb-2">Prêt pour l'analyse</h3>
                <p className="text-steel max-w-sm">
                  Remplissez le formulaire et cliquez sur "Analyser le risque" pour obtenir une prédiction
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default SinglePrediction;
