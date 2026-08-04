import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

// The react-hooks recommended preset is a React Compiler rule set (16 rules).
// This project does not use React Compiler, so the two classic rules are
// registered directly rather than spreading the preset. typescript-eslint is
// here for its parser only — without it, no .ts/.tsx file parses.
export default [
  { ignores: ["dist", "coverage"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: { parser: tseslint.parser },
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "error",
    },
  },
];
