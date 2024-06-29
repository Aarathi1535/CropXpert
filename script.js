document.addEventListener('DOMContentLoaded', function () {
    var tabLinks = document.querySelectorAll('.tab-link');
    var tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(function(link) {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            var targetTab = this.getAttribute('data-tab');

            // Remove 'active' class from all tab contents
            tabContents.forEach(function(content) {
                content.classList.remove('active');
            });

            // Add 'active' class to the target tab content
            document.getElementById(targetTab).classList.add('active');

            // Remove 'active' class from all tab links
            tabLinks.forEach(function(link) {
                link.classList.remove('active');
            });

            // Add 'active' class to the clicked tab link
            this.classList.add('active');
        });
    });
});
