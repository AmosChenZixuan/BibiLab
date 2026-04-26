const DELAYS = [0, 0.3, 0.6];

export function PulseRing() {
  return (
    <span className="relative inline-block w-6 h-6">
      {DELAYS.map((d) => (
        <span
          key={d}
          className="absolute inset-0 rounded-full"
          style={{
            boxShadow: "0 0 0 2px var(--color-muted)",
            animation: "chat-pulse-ring 1.4s ease-in-out infinite",
            animationDelay: `${d}s`,
          }}
        />
      ))}
    </span>
  );
}
