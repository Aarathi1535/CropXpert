document.addEventListener('DOMContentLoaded', function () {
    var tabLinks = document.querySelectorAll('.tab-link');
    var tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(function(link) {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            var targetTab = this.getAttribute('data-tab');

            // Remove 'active' class from all tab contents and tab links
            tabContents.forEach(function(content) {
                content.classList.remove('active');
            });
            tabLinks.forEach(function(tabLink) {
                tabLink.classList.remove('active');
            });

            // Add 'active' class to the target tab content and clicked tab link
            document.getElementById(targetTab).classList.add('active');
            this.classList.add('active');

            // Optional: Smooth scroll to the top of the clicked section
            var offsetTop = document.getElementById(targetTab).offsetTop;
            window.scrollTo({ top: offsetTop, behavior: 'smooth' });
        });
    });
});
