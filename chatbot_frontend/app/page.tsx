"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Send, User, MessageCircle, Bot } from "lucide-react";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

function formatMessageText(text: string) {
  const lines = text.split("\n");
  const formattedLines: React.ReactNode[] = [];

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();
    if (trimmedLine.match(/^#{1,6}\s+/)) {
      const headingText = trimmedLine.replace(/^#{1,6}\s+/, "");
      formattedLines.push(
        <div
          key={index}
          className="font-bold text-gray-900 mt-3 mb-2 first:mt-0"
        >
          {headingText}
        </div>
      );
    } else if (
      trimmedLine.endsWith(":") &&
      trimmedLine.length > 3 &&
      trimmedLine.length < 50 &&
      !trimmedLine.includes("http")
    ) {
      formattedLines.push(
        <div
          key={index}
          className="font-bold text-gray-900 mt-3 mb-1 first:mt-0"
        >
          {trimmedLine}
        </div>
      );
    } else if (
      trimmedLine === trimmedLine.toUpperCase() &&
      trimmedLine.length > 3 &&
      trimmedLine.length < 50 &&
      /^[A-Z\s\d\-_]+$/.test(trimmedLine) &&
      !trimmedLine.includes("HTTP")
    ) {
      formattedLines.push(
        <div
          key={index}
          className="font-bold text-gray-900 mt-3 mb-2 first:mt-0 underline"
        >
          {trimmedLine}
        </div>
      );
    } else if (trimmedLine.match(/^(\d+\.|•|\*|-)\s*[A-Z]/)) {
      const parts = trimmedLine.split(/^(\d+\.|•|\*|-)\s*/);
      if (parts.length >= 3) {
        const bullet = parts[1];
        const content = parts[2];
        if (content.endsWith(":") && content.length < 40) {
          formattedLines.push(
            <div key={index} className="font-bold text-gray-900 mt-2 mb-1">
              {bullet} {content}
            </div>
          );
        } else {
          formattedLines.push(
            <div key={index} className="mt-1">
              <span className="font-semibold">{bullet}</span> {content}
            </div>
          );
        }
      } else {
        formattedLines.push(
          <div key={index} className="mt-1">
            {trimmedLine}
          </div>
        );
      }
    } else if (trimmedLine) {
      formattedLines.push(
        <div key={index} className="mt-1 first:mt-0">
          {trimmedLine}
        </div>
      );
    } else {
      formattedLines.push(<div key={index} className="h-2"></div>);
    }
  });

  return <div>{formattedLines}</div>;
}

function ChatMessage({ message }: { message: Message }) {
  return (
    <div
      className={`flex ${
        message.isUser ? "justify-end" : "justify-start"
      } mb-4 sm:mb-6`}
    >
      {message.isUser ? (
        <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-md">
          <div className="bg-blue-500 text-white px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tr-sm">
            <div className="text-xs sm:text-sm">
              {formatMessageText(message.text)}
            </div>
          </div>
          <div className="bg-blue-500 rounded-full p-1.5 sm:p-2 flex-shrink-0">
            <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
          </div>
        </div>
      ) : (
        <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
          <div className="flex-shrink-0">
            <img
              src="/cylinder-logo.png"
              alt="Gas Cylinder"
              className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
            />
          </div>
          <div>
            <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
              <div className="text-xs sm:text-sm leading-relaxed">
                {formatMessageText(message.text)}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1 ml-3 sm:ml-4">
              {message.timestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                hour12: true,
              })}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
      <div className="flex-shrink-0">
        <img
          src="/cylinder-logo.png"
          alt="Gas Cylinder"
          className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
        />
      </div>
      <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
        <div className="flex space-x-1">
          <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" />
          <div
            className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: "150ms" }}
          />
          <div
            className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: "300ms" }}
          />
        </div>
      </div>
    </div>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! \n I'm WLGA AI Assistant.\n How can I help you today?",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("chat-session-id");
      if (saved) {
        setSessionId(saved);
      } else {
        const newId = crypto.randomUUID();
        localStorage.setItem("chat-session-id", newId);
        setSessionId(newId);
      }
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading || !sessionId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const userInput = input.toLowerCase().trim();
    setInput("");
    setIsLoading(true);

    const whoAreYouPatterns = [
      "who are you",
      "what are you",
      "who r u",
      "what r u",
      "tell me about yourself",
      "introduce yourself",
      "what is your name",
      "whats your name",
    ];

    const isWhoAreYouQuestion = whoAreYouPatterns.some((pattern) =>
      userInput.includes(pattern)
    );

    if (isWhoAreYouQuestion) {
      setTimeout(() => {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: "I'm WLGA AI Assistant  your smart companion powered by advanced language models. I'm here to help you with information, automation, and support  all crafted by the WLGA team.",
          isUser: false,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botMessage]);
        setIsLoading(false);
      }, 1000);
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input, session_id: sessionId }),
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();

      if (!data.response && data.session_id && data.session_id !== sessionId) {
        setMessages([{
        id: "1",
        text: "Hello! I'm WLGA AI Assistant. How can I help you today?",
        isUser: false,
        timestamp: new Date(),
      }]);
        setSessionId(data.session_id);
        localStorage.setItem("chat-session-id", data.session_id);
        setIsLoading(false);
        return;
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text:
          data.response ||
          "I apologize, but I couldn't process your request at the moment.",
        isUser: false,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          text: "Sorry, I'm having trouble connecting to the server. Please try again later.",
          isUser: false,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-purple-700">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-center items-center h-14 sm:h-16">
            <div className="flex items-center space-x-2">
              <div className="bg-gradient-to-r from-purple-500 to-blue-500 rounded-full p-2">
                <Bot className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <span className="text-lg sm:text-xl font-bold text-gray-800">
                WLGA
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex items-center justify-center min-h-[calc(100vh-56px)] sm:min-h-[calc(100vh-64px)] p-2 sm:p-4">
        <div className="w-full max-w-4xl">
          <div className="text-center mb-4 sm:mb-8 px-4">
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-3 mb-3 sm:mb-4">
              <div className="flex items-center space-x-2 sm:space-x-3">
                <MessageCircle className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                <h1 className="text-xl sm:text-3xl font-bold text-white">
                  WLGA AI Assistant
                </h1>
              </div>
              <div className="bg-green-500 text-white px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium">
                Online
              </div>
            </div>
            <p className="text-white/80 text-sm sm:text-lg px-2">
              Intelligent AI Chatbot Platform powered by advanced language
              models
            </p>
          </div>

          <div className="bg-white rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden mx-2 sm:mx-0">
            <div className="h-64 sm:h-96 hide-scrollbar p-3 sm:p-6 space-y-2 sm:space-y-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && <LoadingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            <div className="border-t border-gray-200 p-3 sm:p-6">
              <div className="flex items-center space-x-2 sm:space-x-4">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  className="flex-1 border border-gray-300 rounded-full px-4 py-2 sm:px-6 sm:py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-700 placeholder-gray-500 text-sm sm:text-base"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 sm:px-6 sm:py-3 rounded-full transition-colors duration-200 flex items-center space-x-1 sm:space-x-2 font-medium text-sm sm:text-base"
                >
                  <Send className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="bg-gray-100 py-4 sm:py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap justify-center items-center space-x-4 sm:space-x-8 mb-2 sm:mb-4 text-sm sm:text-base">
            <a
              href="#"
              className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
            >
              Privacy Policy
            </a>
            <a
              href="#"
              className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
            >
              Terms of Service
            </a>
            <a
              href="#"
              className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
            >
              Support
            </a>
            <a
              href="#"
              className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
            >
              API Documentation
            </a>
          </div>
          <div className="text-center text-gray-600 text-sm sm:text-base">
            <p>© 2025 WLGA. Built with ❤️ using advanced AI technology</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

// "use client";

// import type React from "react";
// import { useState, useRef, useEffect } from "react";
// import { Send, User, MessageCircle, Bot } from "lucide-react";

// interface Message {
//   id: string;
//   text: string;
//   isUser: boolean;
//   timestamp: Date;
// }

// function formatMessageText(text: string) {
//   const lines = text.split("\n");
//   const formattedLines: React.ReactNode[] = [];

//   lines.forEach((line, index) => {
//     const trimmedLine = line.trim();
//     if (trimmedLine.match(/^#{1,6}\s+/)) {
//       const headingText = trimmedLine.replace(/^#{1,6}\s+/, "");
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-2 first:mt-0"
//         >
//           {headingText}
//         </div>
//       );
//     } else if (
//       trimmedLine.endsWith(":") &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       !trimmedLine.includes("http")
//     ) {
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-1 first:mt-0"
//         >
//           {trimmedLine}
//         </div>
//       );
//     } else if (
//       trimmedLine === trimmedLine.toUpperCase() &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       /^[A-Z\s\d\-_]+$/.test(trimmedLine) &&
//       !trimmedLine.includes("HTTP")
//     ) {
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-2 first:mt-0 underline"
//         >
//           {trimmedLine}
//         </div>
//       );
//     } else if (trimmedLine.match(/^(\d+\.|•|\*|-)\s*[A-Z]/)) {
//       const parts = trimmedLine.split(/^(\d+\.|•|\*|-)\s*/);
//       if (parts.length >= 3) {
//         const bullet = parts[1];
//         const content = parts[2];
//         if (content.endsWith(":") && content.length < 40) {
//           formattedLines.push(
//             <div key={index} className="font-bold text-gray-900 mt-2 mb-1">
//               {bullet} {content}
//             </div>
//           );
//         } else {
//           formattedLines.push(
//             <div key={index} className="mt-1">
//               <span className="font-semibold">{bullet}</span> {content}
//             </div>
//           );
//         }
//       } else {
//         formattedLines.push(
//           <div key={index} className="mt-1">
//             {trimmedLine}
//           </div>
//         );
//       }
//     } else if (trimmedLine) {
//       formattedLines.push(
//         <div key={index} className="mt-1 first:mt-0">
//           {trimmedLine}
//         </div>
//       );
//     } else {
//       formattedLines.push(<div key={index} className="h-2"></div>);
//     }
//   });

//   return <div>{formattedLines}</div>;
// }

// function ChatMessage({ message }: { message: Message }) {
//   return (
//     <div
//       className={`flex ${
//         message.isUser ? "justify-end" : "justify-start"
//       } mb-4 sm:mb-6`}
//     >
//       {message.isUser ? (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-md">
//           <div className="bg-blue-500 text-white px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tr-sm">
//             <div className="text-xs sm:text-sm">
//               {formatMessageText(message.text)}
//             </div>
//           </div>
//           <div className="bg-blue-500 rounded-full p-1.5 sm:p-2 flex-shrink-0">
//             <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
//           </div>
//         </div>
//       ) : (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//           <div className="flex-shrink-0">
//             <img
//               src="/cylinder-logo.png"
//               alt="Gas Cylinder"
//               className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
//             />
//           </div>
//           <div>
//             <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//               <div className="text-xs sm:text-sm leading-relaxed">
//                 {formatMessageText(message.text)}
//               </div>
//             </div>
//             <p className="text-xs text-gray-500 mt-1 ml-3 sm:ml-4">
//               {message.timestamp.toLocaleTimeString([], {
//                 hour: "2-digit",
//                 minute: "2-digit",
//                 hour12: true,
//               })}
//             </p>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// function LoadingIndicator() {
//   return (
//     <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//       <div className="flex-shrink-0">
//         <img
//           src="/cylinder-logo.png"
//           alt="Gas Cylinder"
//           className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
//         />
//       </div>
//       <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//         <div className="flex space-x-1">
//           <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" />
//           <div
//             className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
//             style={{ animationDelay: "150ms" }}
//           />
//           <div
//             className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
//             style={{ animationDelay: "300ms" }}
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default function ChatInterface() {
//   const [messages, setMessages] = useState<Message[]>([
//     {
//       id: "1",
//       text: "Hello! \n I'm WLGA AI Assistant.\n How can I help you today?",
//       isUser: false,
//       timestamp: new Date(),
//     },
//   ]);
//   const [input, setInput] = useState("");
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const [sessionId, setSessionId] = useState<string | null>(null);

//   useEffect(() => {
//     if (typeof window !== "undefined") {
//       const saved = localStorage.getItem("chat-session-id");
//       if (saved) {
//         setSessionId(saved);
//       } else {
//         const newId = crypto.randomUUID();
//         localStorage.setItem("chat-session-id", newId);
//         setSessionId(newId);
//       }
//     }
//   }, []);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const handleSend = async () => {
//     if (!input.trim() || isLoading || !sessionId) return;

//     const userMessage: Message = {
//       id: Date.now().toString(),
//       text: input,
//       isUser: true,
//       timestamp: new Date(),
//     };

//     setMessages((prev) => [...prev, userMessage]);
//     const userInput = input.toLowerCase().trim();
//     setInput("");
//     setIsLoading(true);

//     const whoAreYouPatterns = [
//       "who are you",
//       "what are you",
//       "who r u",
//       "what r u",
//       "tell me about yourself",
//       "introduce yourself",
//       "what is your name",
//       "whats your name",
//     ];

//     const isWhoAreYouQuestion = whoAreYouPatterns.some((pattern) =>
//       userInput.includes(pattern)
//     );

//     if (isWhoAreYouQuestion) {
//       setTimeout(() => {
//         const botMessage: Message = {
//           id: (Date.now() + 1).toString(),
//           text: "I'm WLGA AI Assistant  your smart companion powered by advanced language models. I'm here to help you with information, automation, and support  all crafted by the WLGA team.",
//           isUser: false,
//           timestamp: new Date(),
//         };
//         setMessages((prev) => [...prev, botMessage]);
//         setIsLoading(false);
//       }, 1000);
//       return;
//     }

//     try {
//       // let sessionId = localStorage.getItem("chat-session-id");
//       // if (!sessionId) {
//       //   sessionId = crypto.randomUUID();
//       //   localStorage.setItem("chat-session-id", sessionId);
//       // }

//       const response = await fetch("http://localhost:5000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ query: input, session_id: sessionId }),
//       });

//       if (!response.ok) throw new Error("Network response was not ok");

//       const data = await response.json();

//       if (!data.response && data.session_id && data.session_id !== sessionId) {
//         setMessages([]);
//         setSessionId(data.session_id);
//         localStorage.setItem("chat-session-id", data.session_id);
//         setIsLoading(false);
//         return;
//       }

//       const botMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         text:
//           data.response ||
//           "I apologize, but I couldn't process your request at the moment.",
//         isUser: false,
//         timestamp: new Date(),
//       };

//       setMessages((prev) => [...prev, botMessage]);
//     } catch (error) {
//       console.error("Error:", error);
//       setMessages((prev) => [
//         ...prev,
//         {
//           id: (Date.now() + 1).toString(),
//           text: "Sorry, I'm having trouble connecting to the server. Please try again later.",
//           isUser: false,
//           timestamp: new Date(),
//         },
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleKeyPress = (e: React.KeyboardEvent) => {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault();
//       handleSend();
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-purple-700">
//       <header className="bg-white shadow-sm">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-center items-center h-14 sm:h-16">
//             <div className="flex items-center space-x-2">
//               <div className="bg-gradient-to-r from-purple-500 to-blue-500 rounded-full p-2">
//                 <Bot className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
//               </div>
//               <span className="text-lg sm:text-xl font-bold text-gray-800">
//                 WLGA
//               </span>
//             </div>
//           </div>
//         </div>
//       </header>

//       <div className="flex items-center justify-center min-h-[calc(100vh-56px)] sm:min-h-[calc(100vh-64px)] p-2 sm:p-4">
//         <div className="w-full max-w-4xl">
//           <div className="text-center mb-4 sm:mb-8 px-4">
//             <div className="flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-3 mb-3 sm:mb-4">
//               <div className="flex items-center space-x-2 sm:space-x-3">
//                 <MessageCircle className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
//                 <h1 className="text-xl sm:text-3xl font-bold text-white">
//                   WLGA AI Assistant
//                 </h1>
//               </div>
//               <div className="bg-green-500 text-white px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium">
//                 Online
//               </div>
//             </div>
//             <p className="text-white/80 text-sm sm:text-lg px-2">
//               Intelligent AI Chatbot Platform powered by advanced language
//               models
//             </p>
//           </div>

//           <div className="bg-white rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden mx-2 sm:mx-0">
//             <div className="h-64 sm:h-96 hide-scrollbar p-3 sm:p-6 space-y-2 sm:space-y-4">
//               {messages.map((message) => (
//                 <ChatMessage key={message.id} message={message} />
//               ))}
//               {isLoading && <LoadingIndicator />}
//               <div ref={messagesEndRef} />
//             </div>

//             <div className="border-t border-gray-200 p-3 sm:p-6">
//               <div className="flex items-center space-x-2 sm:space-x-4">
//                 <input
//                   type="text"
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   onKeyPress={handleKeyPress}
//                   placeholder="Type your message here..."
//                   className="flex-1 border border-gray-300 rounded-full px-4 py-2 sm:px-6 sm:py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-700 placeholder-gray-500 text-sm sm:text-base"
//                   disabled={isLoading}
//                 />
//                 <button
//                   onClick={handleSend}
//                   disabled={!input.trim() || isLoading}
//                   className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 sm:px-6 sm:py-3 rounded-full transition-colors duration-200 flex items-center space-x-1 sm:space-x-2 font-medium text-sm sm:text-base"
//                 >
//                   <Send className="w-3 h-3 sm:w-4 sm:h-4" />
//                   <span className="hidden sm:inline">Send</span>
//                 </button>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>

//       <footer className="bg-gray-100 py-4 sm:py-8">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex flex-wrap justify-center items-center space-x-4 sm:space-x-8 mb-2 sm:mb-4 text-sm sm:text-base">
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Privacy Policy
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Terms of Service
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Support
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               API Documentation
//             </a>
//           </div>
//           <div className="text-center text-gray-600 text-sm sm:text-base">
//             <p>© 2025 WLGA. Built with ❤️ using advanced AI technology</p>
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// }

// "use client";

// import type React from "react";
// import { useState, useRef, useEffect } from "react";
// import { Send, User, MessageCircle, Bot } from "lucide-react";

// interface Message {
//   id: string;
//   text: string;
//   isUser: boolean;
//   timestamp: Date;
// }

// function formatMessageText(text: string) {
//   const lines = text.split("\n");
//   const formattedLines: React.ReactNode[] = [];

//   lines.forEach((line, index) => {
//     const trimmedLine = line.trim();
//     if (trimmedLine.match(/^#{1,6}\s+/)) {
//       const headingText = trimmedLine.replace(/^#{1,6}\s+/, "");
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-2 first:mt-0"
//         >
//           {headingText}
//         </div>
//       );
//     } else if (
//       trimmedLine.endsWith(":") &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       !trimmedLine.includes("http")
//     ) {
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-1 first:mt-0"
//         >
//           {trimmedLine}
//         </div>
//       );
//     } else if (
//       trimmedLine === trimmedLine.toUpperCase() &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       /^[A-Z\s\d\-_]+$/.test(trimmedLine) &&
//       !trimmedLine.includes("HTTP")
//     ) {
//       formattedLines.push(
//         <div
//           key={index}
//           className="font-bold text-gray-900 mt-3 mb-2 first:mt-0 underline"
//         >
//           {trimmedLine}
//         </div>
//       );
//     } else if (trimmedLine.match(/^(\d+\.|•|\*|-)\s*[A-Z]/)) {
//       const parts = trimmedLine.split(/^(\d+\.|•|\*|-)\s*/);
//       if (parts.length >= 3) {
//         const bullet = parts[1];
//         const content = parts[2];
//         if (content.endsWith(":") && content.length < 40) {
//           formattedLines.push(
//             <div key={index} className="font-bold text-gray-900 mt-2 mb-1">
//               {bullet} {content}
//             </div>
//           );
//         } else {
//           formattedLines.push(
//             <div key={index} className="mt-1">
//               <span className="font-semibold">{bullet}</span> {content}
//             </div>
//           );
//         }
//       } else {
//         formattedLines.push(
//           <div key={index} className="mt-1">
//             {trimmedLine}
//           </div>
//         );
//       }
//     } else if (trimmedLine) {
//       formattedLines.push(
//         <div key={index} className="mt-1 first:mt-0">
//           {trimmedLine}
//         </div>
//       );
//     } else {
//       formattedLines.push(<div key={index} className="h-2"></div>);
//     }
//   });

//   return <div>{formattedLines}</div>;
// }

// function ChatMessage({ message }: { message: Message }) {
//   return (
//     <div
//       className={`flex ${
//         message.isUser ? "justify-end" : "justify-start"
//       } mb-4 sm:mb-6`}
//     >
//       {message.isUser ? (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-md">
//           <div className="bg-blue-500 text-white px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tr-sm">
//             <div className="text-xs sm:text-sm">
//               {formatMessageText(message.text)}
//             </div>
//           </div>
//           <div className="bg-blue-500 rounded-full p-1.5 sm:p-2 flex-shrink-0">
//             <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
//           </div>
//         </div>
//       ) : (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//           <div className="flex-shrink-0">
//             <img
//               src="/cylinder-logo.png"
//               alt="Gas Cylinder"
//               className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
//             />
//           </div>
//           <div>
//             <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//               <div className="text-xs sm:text-sm leading-relaxed">
//                 {formatMessageText(message.text)}
//               </div>
//             </div>
//             <p className="text-xs text-gray-500 mt-1 ml-3 sm:ml-4">
//               {message.timestamp.toLocaleTimeString([], {
//                 hour: "2-digit",
//                 minute: "2-digit",
//                 hour12: true,
//               })}
//             </p>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// function LoadingIndicator() {
//   return (
//     <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//       <div className="flex-shrink-0">
//         <img
//           src="/cylinder-logo.png"
//           alt="Gas Cylinder"
//           className="w-8 h-8 sm:w-10 sm:h-10 object-contain"
//         />
//       </div>
//       <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//         <div className="flex space-x-1">
//           <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" />
//           <div
//             className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
//             style={{ animationDelay: "150ms" }}
//           />
//           <div
//             className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce"
//             style={{ animationDelay: "300ms" }}
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default function ChatInterface() {
//   const [messages, setMessages] = useState<Message[]>([
//     {
//       id: "1",
//       text: "Hello! \n I'm WLGA AI Assistant.\n How can I help you today?",
//       isUser: false,
//       timestamp: new Date(),
//     },
//   ]);
//   const [input, setInput] = useState("");
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const handleSend = async () => {
//     if (!input.trim() || isLoading) return;

//     const userMessage: Message = {
//       id: Date.now().toString(),
//       text: input,
//       isUser: true,
//       timestamp: new Date(),
//     };

//     setMessages((prev) => [...prev, userMessage]);
//     const userInput = input.toLowerCase().trim();
//     setInput("");
//     setIsLoading(true);

//     const whoAreYouPatterns = [
//       "who are you",
//       "what are you",
//       "who r u",
//       "what r u",
//       "tell me about yourself",
//       "introduce yourself",
//       "what is your name",
//       "whats your name",
//     ];

//     const isWhoAreYouQuestion = whoAreYouPatterns.some((pattern) =>
//       userInput.includes(pattern)
//     );

//     if (isWhoAreYouQuestion) {
//       setTimeout(() => {
//         const botMessage: Message = {
//           id: (Date.now() + 1).toString(),
//           text: "I'm WLGA AI Assistant  your smart companion powered by advanced language models. I'm here to help you with information, automation, and support  all crafted by the WLGA team.",
//           isUser: false,
//           timestamp: new Date(),
//         };
//         setMessages((prev) => [...prev, botMessage]);
//         setIsLoading(false);
//       }, 1000);
//       return;
//     }

//     try {
//       let sessionId = localStorage.getItem("chat-session-id");
//       if (!sessionId) {
//         sessionId = crypto.randomUUID();
//         localStorage.setItem("chat-session-id", sessionId);
//       }

//       const response = await fetch("http://localhost:5000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ query: input, session_id: sessionId }),
//       });

//       if (!response.ok) throw new Error("Network response was not ok");

//       const data = await response.json();

//       const botMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         text:
//           data.response ||
//           "I apologize, but I couldn't process your request at the moment.",
//         isUser: false,
//         timestamp: new Date(),
//       };

//       setMessages((prev) => [...prev, botMessage]);
//     } catch (error) {
//       console.error("Error:", error);
//       setMessages((prev) => [
//         ...prev,
//         {
//           id: (Date.now() + 1).toString(),
//           text: "Sorry, I'm having trouble connecting to the server. Please try again later.",
//           isUser: false,
//           timestamp: new Date(),
//         },
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleKeyPress = (e: React.KeyboardEvent) => {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault();
//       handleSend();
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-purple-700">
//       <header className="bg-white shadow-sm">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-center items-center h-14 sm:h-16">
//             <div className="flex items-center space-x-2">
//               <div className="bg-gradient-to-r from-purple-500 to-blue-500 rounded-full p-2">
//                 <Bot className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
//               </div>
//               <span className="text-lg sm:text-xl font-bold text-gray-800">
//                 WLGA
//               </span>
//             </div>
//           </div>
//         </div>
//       </header>

//       <div className="flex items-center justify-center min-h-[calc(100vh-56px)] sm:min-h-[calc(100vh-64px)] p-2 sm:p-4">
//         <div className="w-full max-w-4xl">
//           <div className="text-center mb-4 sm:mb-8 px-4">
//             <div className="flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-3 mb-3 sm:mb-4">
//               <div className="flex items-center space-x-2 sm:space-x-3">
//                 <MessageCircle className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
//                 <h1 className="text-xl sm:text-3xl font-bold text-white">
//                   WLGA AI Assistant
//                 </h1>
//               </div>
//               <div className="bg-green-500 text-white px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium">
//                 Online
//               </div>
//             </div>
//             <p className="text-white/80 text-sm sm:text-lg px-2">
//               Intelligent AI Chatbot Platform powered by advanced language
//               models
//             </p>
//           </div>

//           <div className="bg-white rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden mx-2 sm:mx-0">
//             <div className="h-64 sm:h-96 hide-scrollbar p-3 sm:p-6 space-y-2 sm:space-y-4">
//               {messages.map((message) => (
//                 <ChatMessage key={message.id} message={message} />
//               ))}
//               {isLoading && <LoadingIndicator />}
//               <div ref={messagesEndRef} />
//             </div>

//             <div className="border-t border-gray-200 p-3 sm:p-6">
//               <div className="flex items-center space-x-2 sm:space-x-4">
//                 <input
//                   type="text"
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   onKeyPress={handleKeyPress}
//                   placeholder="Type your message here..."
//                   className="flex-1 border border-gray-300 rounded-full px-4 py-2 sm:px-6 sm:py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-700 placeholder-gray-500 text-sm sm:text-base"
//                   disabled={isLoading}
//                 />
//                 <button
//                   onClick={handleSend}
//                   disabled={!input.trim() || isLoading}
//                   className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 sm:px-6 sm:py-3 rounded-full transition-colors duration-200 flex items-center space-x-1 sm:space-x-2 font-medium text-sm sm:text-base"
//                 >
//                   <Send className="w-3 h-3 sm:w-4 sm:h-4" />
//                   <span className="hidden sm:inline">Send</span>
//                 </button>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>

//       <footer className="bg-gray-100 py-4 sm:py-8">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex flex-wrap justify-center items-center space-x-4 sm:space-x-8 mb-2 sm:mb-4 text-sm sm:text-base">
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Privacy Policy
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Terms of Service
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               Support
//             </a>
//             <a
//               href="#"
//               className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0"
//             >
//               API Documentation
//             </a>
//           </div>
//           <div className="text-center text-gray-600 text-sm sm:text-base">
//             <p>© 2025 WLGA. Built with ❤️ using advanced AI technology</p>
//           </div>
//         </div>
//       </footer>
//     </div>
//   );
// }

// "use client"

// import type React from "react"
// import { useState, useRef, useEffect } from "react"
// import { Send, User, MessageCircle, Bot } from "lucide-react"

// interface Message {
//   id: string
//   text: string
//   isUser: boolean
//   timestamp: Date
// }

// function formatMessageText(text: string) {
//   const lines = text.split("\n")
//   const formattedLines: React.ReactNode[] = []

//   lines.forEach((line, index) => {
//     const trimmedLine = line.trim()
//     if (trimmedLine.match(/^#{1,6}\s+/)) {
//       const headingText = trimmedLine.replace(/^#{1,6}\s+/, "")
//       formattedLines.push(
//         <div key={index} className="font-bold text-gray-900 mt-3 mb-2 first:mt-0">{headingText}</div>
//       )
//     } else if (
//       trimmedLine.endsWith(":") &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       !trimmedLine.includes("http")
//     ) {
//       formattedLines.push(
//         <div key={index} className="font-bold text-gray-900 mt-3 mb-1 first:mt-0">{trimmedLine}</div>
//       )
//     } else if (
//       trimmedLine === trimmedLine.toUpperCase() &&
//       trimmedLine.length > 3 &&
//       trimmedLine.length < 50 &&
//       /^[A-Z\s\d\-_]+$/.test(trimmedLine) &&
//       !trimmedLine.includes("HTTP")
//     ) {
//       formattedLines.push(
//         <div key={index} className="font-bold text-gray-900 mt-3 mb-2 first:mt-0 underline">{trimmedLine}</div>
//       )
//     } else if (trimmedLine.match(/^(\d+\.|•|\*|-)\s*[A-Z]/)) {
//       const parts = trimmedLine.split(/^(\d+\.|•|\*|-)\s*/)
//       if (parts.length >= 3) {
//         const bullet = parts[1]
//         const content = parts[2]
//         if (content.endsWith(":") && content.length < 40) {
//           formattedLines.push(
//             <div key={index} className="font-bold text-gray-900 mt-2 mb-1">{bullet} {content}</div>
//           )
//         } else {
//           formattedLines.push(
//             <div key={index} className="mt-1"><span className="font-semibold">{bullet}</span> {content}</div>
//           )
//         }
//       } else {
//         formattedLines.push(<div key={index} className="mt-1">{trimmedLine}</div>)
//       }
//     } else if (trimmedLine) {
//       formattedLines.push(<div key={index} className="mt-1 first:mt-0">{trimmedLine}</div>)
//     } else {
//       formattedLines.push(<div key={index} className="h-2"></div>)
//     }
//   })

//   return <div>{formattedLines}</div>
// }

// function ChatMessage({ message }: { message: Message }) {
//   return (
//     <div className={`flex ${message.isUser ? "justify-end" : "justify-start"} mb-4 sm:mb-6`}>
//       {message.isUser ? (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-md">
//           <div className="bg-blue-500 text-white px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tr-sm">
//             <div className="text-xs sm:text-sm">{formatMessageText(message.text)}</div>
//           </div>
//           <div className="bg-blue-500 rounded-full p-1.5 sm:p-2 flex-shrink-0">
//             <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
//           </div>
//         </div>
//       ) : (
//         <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//           <div className="flex-shrink-0">
//             <img src="/cylinder-logo.png" alt="Gas Cylinder" className="w-8 h-8 sm:w-10 sm:h-10 object-contain" />
//           </div>
//           <div>
//             <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//               <div className="text-xs sm:text-sm leading-relaxed">{formatMessageText(message.text)}</div>
//             </div>
//             <p className="text-xs text-gray-500 mt-1 ml-3 sm:ml-4">
//               {message.timestamp.toLocaleTimeString([], {
//                 hour: "2-digit",
//                 minute: "2-digit",
//                 hour12: true,
//               })}
//             </p>
//           </div>
//         </div>
//       )}
//     </div>
//   )
// }

// function LoadingIndicator() {
//   return (
//     <div className="flex items-start space-x-2 sm:space-x-3 max-w-[280px] sm:max-w-lg">
//       <div className="flex-shrink-0">
//         <img src="/cylinder-logo.png" alt="Gas Cylinder" className="w-8 h-8 sm:w-10 sm:h-10 object-contain" />
//       </div>
//       <div className="bg-gray-100 text-gray-800 px-3 py-2 sm:px-4 sm:py-3 rounded-2xl rounded-tl-sm">
//         <div className="flex space-x-1">
//           <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" />
//           <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
//           <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
//         </div>
//       </div>
//     </div>
//   )
// }

// export default function ChatInterface() {
//   const [messages, setMessages] = useState<Message[]>([
//     {
//       id: "1",
//       text: "Hello! I'm WLGA AI Assistant. How can I help you today?",
//       isUser: false,
//       timestamp: new Date(),
//     },
//   ])
//   const [input, setInput] = useState("")
//   const [isLoading, setIsLoading] = useState(false)
//   const messagesEndRef = useRef<HTMLDivElement>(null)

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
//   }

//   useEffect(() => {
//     scrollToBottom()
//   }, [messages])

//   const handleSend = async () => {
//     if (!input.trim() || isLoading) return

//     const userMessage: Message = {
//       id: Date.now().toString(),
//       text: input,
//       isUser: true,
//       timestamp: new Date(),
//     }

//     setMessages((prev) => [...prev, userMessage])
//     const userInput = input.toLowerCase().trim()
//     setInput("")
//     setIsLoading(true)

//     const whoAreYouPatterns = [
//       "who are you", "what are you", "who r u", "what r u",
//       "tell me about yourself", "introduce yourself", "what is your name", "whats your name"
//     ]

//     const isWhoAreYouQuestion = whoAreYouPatterns.some((pattern) => userInput.includes(pattern))

//     if (isWhoAreYouQuestion) {
//       setTimeout(() => {
//         const botMessage: Message = {
//           id: (Date.now() + 1).toString(),
//           text: "I'm WLGA AI Assistant 🤖  your smart companion powered by advanced language models. I'm here to help you with information, automation, and support  all crafted by the WLGA team.",
//           isUser: false,
//           timestamp: new Date(),
//         }
//         setMessages((prev) => [...prev, botMessage])
//         setIsLoading(false)
//       }, 1000)
//       return
//     }

//     try {
//       const response = await fetch("http://localhost:5000/chat", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ query: input }),
//       })

//       if (!response.ok) throw new Error("Network response was not ok")

//       const data = await response.json()

//       const botMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         text: data.response || "I apologize, but I couldn't process your request at the moment.",
//         isUser: false,
//         timestamp: new Date(),
//       }

//       setMessages((prev) => [...prev, botMessage])
//     } catch (error) {
//       console.error("Error:", error)
//       setMessages((prev) => [
//         ...prev,
//         {
//           id: (Date.now() + 1).toString(),
//           text: "Sorry, I'm having trouble connecting to the server. Please try again later.",
//           isUser: false,
//           timestamp: new Date(),
//         },
//       ])
//     } finally {
//       setIsLoading(false)
//     }
//   }

//   const handleKeyPress = (e: React.KeyboardEvent) => {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault()
//       handleSend()
//     }
//   }

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-purple-700">
//       <header className="bg-white shadow-sm">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-center items-center h-14 sm:h-16">
//             <div className="flex items-center space-x-2">
//               <div className="bg-gradient-to-r from-purple-500 to-blue-500 rounded-full p-2">
//                 <Bot className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
//               </div>
//               <span className="text-lg sm:text-xl font-bold text-gray-800">WLGA</span>
//             </div>
//           </div>
//         </div>
//       </header>

//       <div className="flex items-center justify-center min-h-[calc(100vh-56px)] sm:min-h-[calc(100vh-64px)] p-2 sm:p-4">
//         <div className="w-full max-w-4xl">
//           <div className="text-center mb-4 sm:mb-8 px-4">
//             <div className="flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-3 mb-3 sm:mb-4">
//               <div className="flex items-center space-x-2 sm:space-x-3">
//                 <MessageCircle className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
//                 <h1 className="text-xl sm:text-3xl font-bold text-white">WLGA AI Assistant</h1>
//               </div>
//               <div className="bg-green-500 text-white px-2 py-1 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-medium">
//                 Online
//               </div>
//             </div>
//             <p className="text-white/80 text-sm sm:text-lg px-2">
//               Intelligent AI Chatbot Platform powered by advanced language models
//             </p>
//           </div>

//           <div className="bg-white rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden mx-2 sm:mx-0">
//             <div className="h-64 sm:h-96 hide-scrollbar p-3 sm:p-6 space-y-2 sm:space-y-4">
//               {messages.map((message) => (
//                 <ChatMessage key={message.id} message={message} />
//               ))}
//               {isLoading && <LoadingIndicator />}
//               <div ref={messagesEndRef} />
//             </div>

//             <div className="border-t border-gray-200 p-3 sm:p-6">
//               <div className="flex items-center space-x-2 sm:space-x-4">
//                 <input
//                   type="text"
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   onKeyPress={handleKeyPress}
//                   placeholder="Type your message here..."
//                   className="flex-1 border border-gray-300 rounded-full px-4 py-2 sm:px-6 sm:py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-700 placeholder-gray-500 text-sm sm:text-base"
//                   disabled={isLoading}
//                 />
//                 <button
//                   onClick={handleSend}
//                   disabled={!input.trim() || isLoading}
//                   className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 sm:px-6 sm:py-3 rounded-full transition-colors duration-200 flex items-center space-x-1 sm:space-x-2 font-medium text-sm sm:text-base"
//                 >
//                   <Send className="w-3 h-3 sm:w-4 sm:h-4" />
//                   <span className="hidden sm:inline">Send</span>
//                 </button>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>

//       <footer className="bg-gray-100 py-4 sm:py-8">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex flex-wrap justify-center items-center space-x-4 sm:space-x-8 mb-2 sm:mb-4 text-sm sm:text-base">
//             <a href="#" className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0">Privacy Policy</a>
//             <a href="#" className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0">Terms of Service</a>
//             <a href="#" className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0">Support</a>
//             <a href="#" className="text-gray-600 hover:text-gray-900 mb-2 sm:mb-0">API Documentation</a>
//           </div>
//           <div className="text-center text-gray-600 text-sm sm:text-base">
//             <p>© 2025 WLGA. Built with ❤️ using advanced AI technology</p>
//           </div>
//         </div>
//       </footer>
//     </div>
//   )
// }
