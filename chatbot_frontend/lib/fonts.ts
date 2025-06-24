import { Orbitron } from "next/font/google"

/**
 * Centralised font export – import { orbitron } from "@/lib/fonts"
 * if you need the className elsewhere.
 */
export const orbitron = Orbitron({
  subsets: ["latin"],
  weight: ["400", "700", "900"],
})
