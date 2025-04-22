import React from "react";
import PredictionForm from "./PredictionForm";
import { motion } from "framer-motion";
import { FaHeartbeat, FaBrain, FaLightbulb } from "react-icons/fa";

const HeroSection = () => {
  return (
    <section className="relative py-28 px-6">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-4xl mx-auto text-center"
      >
        <h1 className="text-5xl font-extrabold text-slate-800 mb-6 leading-tight">
          Smarter Readmission Risk Detection
        </h1>
        <p className="text-lg text-slate-600 mb-8 max-w-2xl mx-auto">
          Leverage our AI-powered platform to predict hospital readmissions in
          diabetic patients — helping clinicians make proactive, data-driven
          decisions.
        </p>
      </motion.div>
    </section>
  );
};

const InfoCards = () => {
  const cards = [
    {
      icon: <FaHeartbeat className="text-red-500 text-3xl" />,
      title: "Why Readmissions Matter",
      text: "Preventing early readmissions improves patient outcomes and reduces healthcare costs.",
    },
    {
      icon: <FaBrain className="text-blue-500 text-3xl" />,
      title: "How Our AI Helps",
      text: "Our machine learning model evaluates complex factors to predict patient risk.",
    },
    {
      icon: <FaLightbulb className="text-yellow-500 text-3xl" />,
      title: "Your Next Steps",
      text: "Use insights to tailor discharge planning and follow-up care effectively.",
    },
  ];

  return (
    <section className="pb-20 px-6">
      <div className="max-w-6xl mx-auto grid gap-8 grid-cols-1 md:grid-cols-3">
        {cards.map((card, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2, duration: 0.6 }}
            viewport={{ once: true }}
            className="flex flex-col justify-between bg-white rounded-xl shadow-md p-6 text-center hover:shadow-xl transition"
          >
            <div className="flex items-center justify-center gap-2 mb-4">
              {card.icon}
              <h3 className="text-lg font-bold text-slate-800">{card.title}</h3>
            </div>
            <p className="text-sm text-slate-600">{card.text}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
};

const Footer = () => {
  return (
    <footer className="bg-slate-100 py-6 mt-20 shadow-inner">
      <div className="max-w-6xl mx-auto text-center text-slate-500 text-sm">
        <p>© 2025 Readmission Prediction Tool. All rights reserved.</p>
      </div>
    </footer>
  );
};

const LandingPage = () => {
  return (
    <main className="bg-white min-h-screen font-sans scroll-smooth">
      <div className=" bg-gradient-to-br from-blue-100 to-white">
        <HeroSection />
        <InfoCards />
      </div>
      <section id="prediction" className="scroll-mt-10">
        <PredictionForm />
      </section>
      <Footer />
    </main>
  );
};

export default LandingPage;
