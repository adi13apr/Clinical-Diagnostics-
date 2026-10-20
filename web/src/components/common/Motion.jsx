import { motion } from "framer-motion";

/** Subtle fade-and-rise for cards. Reduced-motion users get no movement (MotionConfig in main.jsx). */
export function FadeIn({ children, delay = 0, className }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut", delay }}
    >
      {children}
    </motion.div>
  );
}
