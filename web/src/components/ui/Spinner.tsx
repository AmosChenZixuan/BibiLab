

export function Spinner({ label }: { label?: string } = {}) {
  return (
      <span className="inline-flex items-center justify-center" role="status" aria-label={label}>
      <span
          className="h-4 w-4 animate-spin rounded-full border-2 border-blue/25 border-t-blue"
          aria-hidden="true"
      />
      </span>
  );
}
