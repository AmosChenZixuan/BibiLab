# Web UI

React + TypeScript SPA. Managed with `npm`.

## Commands

```bash
npm install              # Install frontend dependencies
npm run dev              # Start Vite dev server on :5173
npm run build            # Production build to web/dist
npm run test             # Frontend test suite
npm run test -- list-detail-page   # Focused frontend tests
npm run lint             # Type-check the frontend
```

## Code Layout — `src/`

```
components/ui/    — primitive components (Button, Modal, Panel, Select, Spinner, StatusChip, etc.)
components/*/     — feature components (lists/, lists/lab/, jobs/, layout/, settings/)
pages/            — route-level page components
lib/              — typed api client, types, download helpers, health check
app/              — router, language context
test/             — Vitest test files + setup
```

## Routes

| Path | View |
|---|---|
| `/` | Home: grid of lists |
| `/lists/:id` | List detail: Sources \| Chat \| Lab |
| `/settings` | Global config, health, accounts |

## Conventions

- **Files**: `PascalCase` for components, `kebab-case` for utilities
- **Components**: props interface above component, `ComponentPropsWithoutRef<"tag">` for root element props, `Record<Variant, string>` for variant maps
- **Handlers**: named `handle{Action}`; event props use `on{Action}` prefix (`onDelete`, `onCreate`)
- **State**: `useState` with `set` prefix; async operations use `let cancelled = false` guard in `useEffect`
- **Imports**: use `@/*` alias (`@/components/ui`, `@/lib/api`, `@/lib/types`)
- **API client**: single `api` object in `lib/api.ts` with typed `request<T>` wrapper; errors thrown as `ApiError`
- **Styling**: Tailwind utility classes only; no CSS modules. Inline `style` only for dynamic computed values (widths, positions, URLs). No arbitrary bracket values (e.g. `mt-[10px]`) — use Tailwind's built-in scale or CSS custom properties from `src/styles/app.css` (`--color-*`, `--z-*`, `--font-*`)
