/* 
Author: Andy Yang
Date: 2024-10-11
Project: VisuAI Front End
Description: Created the Visuals and Details of VisuAI Camera with Flask, HTML, CSS, and JavaScript
*/

document.addEventListener('DOMContentLoaded', () => {
    const descriptionDiv = document.getElementById("description");

    async function fetchDescription() {
        try {
            const response = await fetch('/get_description');
            const data = await response.text();
            descriptionDiv.innerText = data;
        } catch (error) {
            console.error("Error fetching description:", error);
        }
    }

    // Fetch descriptions every 5 seconds
    setInterval(fetchDescription, 5000);
});

