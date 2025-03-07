const wrapper = document.querySelector('.wrapper');
const registerLink = document.querySelector('.register-link');
const loginLink = document.querySelector('.login-link');
registerLink.addEventListener('click', (e) => {
    e.preventDefault(); 
    console.log("Register link clicked!");
    wrapper.classList.add('active');
});

loginLink.addEventListener('click', (e) => {
    e.preventDefault(); 
    console.log("Login link clicked!");
    wrapper.classList.remove('active');
});

registerLink.addEventListener('click', (e) => {
    e.preventDefault(); 
    wrapper.classList.add('active');
});

loginLink.addEventListener('click', (e) => {
    e.preventDefault(); 
    wrapper.classList.remove('active');
});
