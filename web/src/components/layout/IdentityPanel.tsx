type IdentityPanelProps = {
  onClose: () => void;
};

const PLATFORMS = [{ key: "bilibili", label: "Bilibili", icon: "B" }];

export default function IdentityPanel({ onClose }: IdentityPanelProps) {
  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-[199] border-0 bg-transparent"
        aria-label="Close identity panel"
        onClick={onClose}
      />
      <div
        className="fixed top-[52px] right-[clamp(16px,3vw,48px)] z-[200] min-w-[180px] rounded-[14px] border border-[rgba(106,147,198,0.12)] bg-[rgba(255,255,255,0.96)] p-4 shadow-[0_8px_32px_rgba(0,0,0,0.12)] backdrop-blur-[18px]"
        role="menu"
        aria-label="Identity"
      >
        <div className="flex flex-wrap gap-3">
          {PLATFORMS.map((platform) => (
            <div key={platform.key} className="flex w-[72px] flex-col items-center gap-1">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-[10px] bg-[rgba(240,139,185,0.14)] font-bold text-[#5b7faa]"
                aria-hidden="true"
              >
                {platform.icon}
              </span>
              <span className="text-[0.72rem] font-semibold text-[#274970]">{platform.label}</span>
              <span className="text-center text-[0.72rem] text-[#8096b3]">Not signed in</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
