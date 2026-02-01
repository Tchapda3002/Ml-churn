import { motion } from 'framer-motion';
import { Activity, Shield, TrendingUp } from 'lucide-react';

const Header = ({ healthStatus, activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'single', label: 'Prédiction', icon: TrendingUp },
    { id: 'batch', label: 'Batch', icon: Activity },
    { id: 'validate', label: 'Validation', icon: Shield },
  ];

  return (
    <header className="glass-dark sticky top-0 z-50 border-b border-graphite/30">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-4"
          >
            <div className="relative">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-gold-dark via-gold to-gold-light flex items-center justify-center gold-glow">
                <TrendingUp className="w-6 h-6 text-midnight" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-risk-low border-2 border-obsidian" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-pearl">ChurnVision</h1>
              <p className="text-sm text-steel">Prédiction de départ client</p>
            </div>
          </motion.div>

          {/* Navigation */}
          <nav className="flex items-center gap-2">
            {tabs.map((tab, index) => (
              <motion.button
                key={tab.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setActiveTab(tab.id)}
                className={`relative flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium transition-all duration-300 ${
                  activeTab === tab.id
                    ? 'text-midnight'
                    : 'text-silver hover:text-pearl hover:bg-slate/30'
                }`}
              >
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-gradient-to-r from-gold-dark via-gold to-gold-light rounded-lg"
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <tab.icon className="w-4 h-4 relative z-10" />
                <span className="relative z-10">{tab.label}</span>
              </motion.button>
            ))}
          </nav>

          {/* Health Status */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-3"
          >
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
              healthStatus === 'healthy'
                ? 'bg-risk-low/10 border border-risk-low/30'
                : healthStatus === 'checking'
                ? 'bg-gold/10 border border-gold/30'
                : 'bg-risk-critical/10 border border-risk-critical/30'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                healthStatus === 'healthy'
                  ? 'bg-risk-low animate-pulse'
                  : healthStatus === 'checking'
                  ? 'bg-gold animate-pulse-gold'
                  : 'bg-risk-critical'
              }`} />
              <span className={`text-sm font-medium ${
                healthStatus === 'healthy'
                  ? 'text-risk-low'
                  : healthStatus === 'checking'
                  ? 'text-gold'
                  : 'text-risk-critical'
              }`}>
                {healthStatus === 'healthy' ? 'API Active' : healthStatus === 'checking' ? 'Connexion...' : 'API Hors ligne'}
              </span>
            </div>
          </motion.div>
        </div>
      </div>
    </header>
  );
};

export default Header;
