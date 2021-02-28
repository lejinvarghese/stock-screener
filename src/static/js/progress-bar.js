function pollIoTUpdates() {
    $.ajax({
        url: "https://api.thingspeak.com/channels/974339/fields/1.json?api_key=HUVMQDSDBY42EM1S&results=1",
        success: function(data) {
            console.log(data)
            if (data) {
                var feed = data["feeds"]
                if (feed) {
                    var latest_entry = feed[0]
                    var weight = +latest_entry["field1"]
                    var progress_level = 100 * (weight) / (0.52217)
                    $('#tomato-bar').attr('aria-valuenow', `${progress_level}%`).css('width', `${progress_level}%`);
                }
            }
            setTimeout(pollIoTUpdates, 2000);
        },
        error: function(e) {
            console.log(e)
            setTimeout(pollIoTUpdates, 2000);
        },
        dataType: "json"
    });

}
pollIoTUpdates()

$(document).ready(function() {

    var all_progress_bars = $(".continuum")
    var screen_width = screen.width;
    for (var i = 0; i < all_progress_bars.length; i++) {
        let progress_bar = all_progress_bars[i];
        let number_of_bars = (screen_width - 50) / 40;
        for (var j = 0; j < number_of_bars; j++) {
            //$(progress_bar).append('<div class="continuum - bar"></div>')
        }
    }
});