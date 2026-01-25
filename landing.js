const revealElements = document.querySelectorAll(".reveal");
const slideElements = document.querySelectorAll(".reveal-slide");

// Smaller margins for mobile
const isMobile = window.innerWidth < 768;

const observerOptions = {
  threshold: 0.05,
  rootMargin: isMobile ? "0px 0px -30px 0px" : "0px 0px -100px 0px",
};

const slideObserverOptions = {
  threshold: 0.05,
  rootMargin: isMobile ? "0px 0px -50px 0px" : "0px 0px -150px 0px",
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("in-view");
      observer.unobserve(entry.target);
    }
  });
}, observerOptions);

const slideObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("in-view");
      slideObserver.unobserve(entry.target);
    }
  });
}, slideObserverOptions);

revealElements.forEach((el) => observer.observe(el));
slideElements.forEach((el) => slideObserver.observe(el));

// Fallback: trigger hero animation on load
window.addEventListener("load", () => {
  const hero = document.querySelector(".hero");
  if (hero) {
    hero.classList.add("in-view");
  }
});
