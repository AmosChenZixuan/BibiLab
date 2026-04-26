interface PulseRingProps {
  className?: string;
}

export function PulseRing({ className }: PulseRingProps) {
  return (
    <span className={`chat-pulse-ring ${className ?? ""}`}>
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
    </span>
  );
}
