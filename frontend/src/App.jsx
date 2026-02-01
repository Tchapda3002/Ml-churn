import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import SinglePrediction from './components/SinglePrediction';
import BatchPrediction from './components/BatchPrediction';
import ValidationMode from './components/ValidationMode';
import { checkHealth } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [healthStatus, setHealthStatus] = useState('checking');

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const response = await checkHealth();
        setHealthStatus(response.status === 'healthy' ? 'healthy' : 'unhealthy');
      } catch (error) {
        setHealthStatus('unhealthy');
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const pageVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  };

  return (
    <div className="min-h-screen">
      <div className="noise-overlay" />

      {/* Ambient background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] rounded-full bg-gold/5 blur-[120px]" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-accent-purple/5 blur-[100px]" />
      </div>

      <Header healthStatus={healthStatus} activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="relative z-10 px-6 py-10">
        <AnimatePresence mode="wait">
          {activeTab === 'single' && (
            <motion.div
              key="single"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3 }}
            >
              <SinglePrediction />
            </motion.div>
          )}

          {activeTab === 'batch' && (
            <motion.div
              key="batch"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3 }}
            >
              <BatchPrediction />
            </motion.div>
          )}

          {activeTab === 'validate' && (
            <motion.div
              key="validate"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.3 }}
            >
              <ValidationMode />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-graphite/30 py-6 mt-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-steel text-sm">
            ChurnVision - Prédiction de départ client
          </p>
          <div className="flex items-center gap-2">
            <span className="text-steel text-sm">Propulsé par</span>
            <span className="text-gradient-gold font-semibold">Machine Learning</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
