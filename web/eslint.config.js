import parser from "@typescript-eslint/parser";
import reactHooks from "eslint-plugin-react-hooks";

// The react-hooks recommended preset is a React Compiler rule set. This project
// does not use React Compiler, so the two classic rules are registered directly
// rather than spreading the preset.
export default [
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: { parser },
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "error",
    },
  },
];
