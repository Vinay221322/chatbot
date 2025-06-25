import { NextResponse } from "next/server"

export async function POST(request: Request) {
  // ---------- parse user text ----------
  const { query } = (await request.json()) as { query?: string }

  // ---------- simple demo reply ----------
  // In a real project you'd call your real backend / LLM here
  let reply = "Sorry, I couldn't understand that. Can you rephrase?"

  // Basic canned response for "who are you"
  if (typeof query === "string") {
    const normalized = query.toLowerCase().trim()
    if (
      [
        "who are you",
        "what are you",
        "who r u",
        "what r u",
        "tell me about yourself",
        "introduce yourself",
        "what is your name",
        "whats your name",
      ].some((p) => normalized.includes(p))
    ) {
      reply =
        "I'm WLGA AI Assistant 🤖  your smart companion powered by advanced language models. I'm here to help you with information, automation, and support — all crafted by the WLGA team."
    } else {
      // echo for demo
      reply = `You said: "${query}"`
    }
  }

  return NextResponse.json({ response: reply })
}
