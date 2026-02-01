import { motion } from 'framer-motion';

const RiskGauge = ({ probability, riskLevel, size = 'large' }) => {
  const percentage = Math.round(probability * 100);

  const riskColors = {
    Low: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', border: 'rgba(16, 185, 129, 0.3)' },
    Medium: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', border: 'rgba(245, 158, 11, 0.3)' },
    High: { color: '#f97316', bg: 'rgba(249, 115, 22, 0.1)', border: 'rgba(249, 115, 22, 0.3)' },
    Critical: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', border: 'rgba(239, 68, 68, 0.3)' },
  };

  const currentRisk = riskColors[riskLevel] || riskColors.Low;
  const isLarge = size === 'large';
  const gaugeSize = isLarge ? 200 : 120;
  const strokeWidth = isLarge ? 12 : 8;
  const radius = (gaugeSize - strokeWidth) / 2;
  const circumference = radius * Math.PI;

  return (
    <div className={`flex flex-col items-center ${isLarge ? 'gap-4' : 'gap-2'}`}>
      <div className="relative" style={{ width: gaugeSize, height: gaugeSize / 2 + 20 }}>
        <svg
          width={gaugeSize}
          height={gaugeSize / 2 + 20}
          viewBox={`0 0 ${gaugeSize} ${gaugeSize / 2 + 20}`}
          className="overflow-visible"
        >
          {/* Background arc */}
          <path
            d={`M ${strokeWidth / 2} ${gaugeSize / 2 + 10} A ${radius} ${radius} 0 0 1 ${gaugeSize - strokeWidth / 2} ${gaugeSize / 2 + 10}`}
            fill="none"
            stroke="rgba(100, 116, 139, 0.2)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Colored arc */}
          <motion.path
            d={`M ${strokeWidth / 2} ${gaugeSize / 2 + 10} A ${radius} ${radius} 0 0 1 ${gaugeSize - strokeWidth / 2} ${gaugeSize / 2 + 10}`}
            fill="none"
            stroke={currentRisk.color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: probability }}
            transition={{ duration: 1.5, ease: 'easeOut', delay: 0.3 }}
            style={{
              filter: `drop-shadow(0 0 8px ${currentRisk.color}40)`,
            }}
          />

          {/* Tick marks */}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const angle = Math.PI * (1 - tick);
            const innerRadius = radius - strokeWidth / 2 - 8;
            const outerRadius = radius - strokeWidth / 2 - 4;
            const x1 = gaugeSize / 2 + innerRadius * Math.cos(angle);
            const y1 = gaugeSize / 2 + 10 - innerRadius * Math.sin(angle);
            const x2 = gaugeSize / 2 + outerRadius * Math.cos(angle);
            const y2 = gaugeSize / 2 + 10 - outerRadius * Math.sin(angle);

            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="rgba(100, 116, 139, 0.5)"
                strokeWidth={2}
              />
            );
          })}
        </svg>

        {/* Center value */}
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-2">
          <motion.span
            className={`font-mono font-bold ${isLarge ? 'text-4xl' : 'text-2xl'}`}
            style={{ color: currentRisk.color }}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
          >
            {percentage}%
          </motion.span>
        </div>
      </div>

      {/* Risk level badge */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className={`px-4 py-2 rounded-full font-semibold ${isLarge ? 'text-base' : 'text-sm'}`}
        style={{
          backgroundColor: currentRisk.bg,
          border: `1px solid ${currentRisk.border}`,
          color: currentRisk.color,
        }}
      >
        Risque {riskLevel}
      </motion.div>
    </div>
  );
};

export default RiskGauge;
