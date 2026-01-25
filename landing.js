const revealElements = document.querySelectorAll(".reveal");
const slideElements = document.querySelectorAll(".reveal-slide");

const observerOptions = {
  threshold: 0.15,
};

const slideObserverOptions = {
  threshold: 0.25,
  rootMargin: "0px 0px -50px 0px",
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
