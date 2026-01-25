const revealElements = document.querySelectorAll(".reveal");
const slideElements = document.querySelectorAll(".reveal-slide");

const observerOptions = {
  threshold: 0.1,
  rootMargin: "0px 0px -100px 0px",
};

const slideObserverOptions = {
  threshold: 0.15,
  rootMargin: "0px 0px -150px 0px",
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
