import { createContext, useContext } from "react";

type NavTitleContextType = HTMLElement | null;

export const NavTitleContext = createContext<NavTitleContextType>(null);

export function useNavTitleContext() {
  return useContext(NavTitleContext);
}
