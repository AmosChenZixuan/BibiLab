

export function Spinner({ label }: { label: string }) {
  return (
      <span className="inline-flex items-center justify-center" role="status" aria-label={label}>
      <span
          className="h-4 w-4 animate-spin rounded-full border-2 border-meta-blue/25 border-t-meta-blue"
          aria-hidden="true"
      />
      </span>
  );
}
