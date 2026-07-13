import { useEffect, useState } from "react";

/** Floating dust particles — the quiet, sunlit-library atmosphere. */
export function DustParticles({ count = 26 }: { count?: number }) {
  const [particles, setParticles] = useState<
    { left: number; top: number; size: number; delay: number; duration: number }[]
  >([]);

  useEffect(() => {
    setParticles(
      Array.from({ length: count }).map(() => ({
        left: Math.random() * 100,
        top: Math.random() * 100,
        size: 1 + Math.random() * 2.5,
        delay: Math.random() * 8,
        duration: 8 + Math.random() * 10,
      })),
    );
  }, [count]);

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      {particles.map((p, i) => (
        <span
          key={i}
          className="absolute rounded-full bg-gold/70"
          style={{
            left: `${p.left}%`,
            top: `${p.top}%`,
            width: p.size,
            height: p.size,
            filter: "blur(0.3px)",
            animation: `float-dust ${p.duration}s ${p.delay}s infinite ease-in-out`,
          }}
        />
      ))}
    </div>
  );
}

/** Slow warm light rays sweeping across a scene. */
export function LightRays() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <div
        className="absolute -top-1/3 left-1/4 h-[160%] w-1/3 rotate-12 bg-gradient-to-b from-gold/25 to-transparent blur-3xl"
        style={{ animation: "light-sweep 14s ease-in-out infinite" }}
      />
      <div
        className="absolute -top-1/3 right-1/5 h-[160%] w-1/4 rotate-12 bg-gradient-to-b from-gold/15 to-transparent blur-3xl"
        style={{ animation: "light-sweep 18s ease-in-out infinite reverse" }}
      />
    </div>
  );
}
