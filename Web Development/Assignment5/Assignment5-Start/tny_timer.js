"use strict";
/*
   New Perspectives on HTML5 and CSS3, 7th Edition
   Tutorial 9
   Review Assignment

   Event Timer
   Author: Yahe Yang
   Date:   Mar/28/2024

*/

// Call showClock() per second
showClock();
setInterval(showClock, 1000);

function showClock() {
  // a. Declare a variable
  var thisDay = new Date();

  // b. Declare variables for local date and time
  var localDate = thisDay.toLocaleDateString();
  var localTime = thisDay.toLocaleTimeString();

  // c. Update the inner HTML of the element with the ID currentTime
  document.getElementById("currentTime").innerHTML =
    "<span>" + localDate + "</span><span>" + localTime + "</span>";

  // d. Call the nextJuly4() function and store the result in j4Date
  var j4Date = nextJuly4(thisDay);

  // e. Set the hours to 9 p.m. (21 in 24-hour format)
  j4Date.setHours(21, 0, 0); // Sets the time to 9:00:00 PM

  // f. Calculate the time difference
  var diff = j4Date.getTime() - thisDay.getTime();
  var days = Math.floor(diff / (1000 * 60 * 60 * 24));
  var hrs = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  var mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  var secs = Math.floor((diff % (1000 * 60)) / 1000);

  // g. Update the text content of the countdown elements
  document.getElementById("dLeft").textContent = days;
  document.getElementById("hLeft").textContent = hrs;
  document.getElementById("mLeft").textContent = mins;
  document.getElementById("sLeft").textContent = secs;
}

function nextJuly4(currentDate) {
  var cYear = currentDate.getFullYear();
  var jDate = new Date("July 4, 2018");
  jDate.setFullYear(cYear);
  if (jDate - currentDate < 0) jDate.setFullYear(cYear + 1);
  return jDate;
}
