type IdentityPanelProps = {
  onClose: () => void;
};

const PLATFORMS = [{ key: "bilibili", label: "Bilibili", icon: "B" }];

export default function IdentityPanel({ onClose }: IdentityPanelProps) {
  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-backdrop border-0 bg-transparent"
        aria-label="Close identity panel"
        onClick={onClose}
      />
      <div
        className="fixed top-[52px] /* below navbar */ right-[clamp(16px,3vw,48px)] z-overlay min-w-[180px] rounded-overlay border border-border bg-white/96 p-4 shadow-overlay backdrop-blur-[18px] /* glass blur — no matching Tailwind step */"
        role="menu"
        aria-label="Identity"
      >
        <div className="flex flex-wrap gap-3">
          {PLATFORMS.map((platform) => (
            <div key={platform.key} className="flex w-[72px] flex-col items-center gap-1">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-icon bg-pink/14 font-bold text-blue"
                aria-hidden="true"
              >
                {platform.icon}
              </span>
              <span className="text-xs font-semibold text-ink">{platform.label}</span>
              <span className="text-center text-xs text-muted">Not signed in</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
