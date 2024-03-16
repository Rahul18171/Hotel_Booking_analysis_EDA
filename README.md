# Hotel Booking Analysis EDA
To discuss the analysis of given hotel bookings data set from 2015-2017. We’ll be doing analysis of given data set in following ways :

• Univariate analysis 

• Hotel wise analysis

• Distribution Channel wise analysis 

• Booking cancellation analysis 

• Timewise analysis

By doing this we’ll try to find out key factors driving the hotel bookings trends.

# Data Summary

Given data set has different columns of variables crucial for hotel bookings.
Some of them are:

hotel: : The category of hotels, which are two resort hotel and city hotel.

is_cancelled : The value of column show the cancellation type. If the booking was
cancelled or not. Values[0,1], where 0 indicates not cancelled.

lead_time : The time between reservation and actual arrival.

stayed_in_weekend_nights:  The number of weekend nights stay per reservation.

stayed_in_weekday_nights:  The number of weekday nights stay per reservation.

meal: Meal preferences per reservation.[BB,FB,HB,SC,Undefined]

Country: The origin country of guest.

market_segment: This column show how reservation was made and what is the
Purpose of reservation. Eg, corporate means corporate trip, TA for travel agency.

distribution_channel: The medium through booking was
market_segment: This column show how reservation was made and what is the
Purpose of reservation. Eg, corporate means corporate trip, TA for travel agency.

distribution_channel: The medium through booking was
made.[Direct,Corporate,TA/TO,undefined,GDS.].

Is_repeated_guest: Shows if the guest is who has arrived earlier or
not.Values[0,1]-->0 indicates no and 1 indicated yes person is repeated guest.

days_in_waiting_list: Number of days between actual booking and transact.

customer_type: Type of customers( Transient, group, etc.

Is_repeated_guest: Shows if the guest is who has arrived earlier or
not.Values[0,1]-->0 indicates no and 1 indicated yes person is repeated guest.

days_in_waiting_list: Number of days between actual booking and transact.

customer_type: Type of customers( Transient, group, etc.)

# Data Summary

![image](https://github.com/Rahul18171/Hotel_Booking_analysis_EDA/assets/130995317/a948d157-3f8e-40c4-86a8-b0bf2b9ed27d)

# Conclusion
● Around 60% bookings are for City hotel and 40% bookings are for Resort hotel, therefore City Hotel is busier than Resort hotel. Also the overall adr of City hotel is slightly higher than Resort hotel. 

● Mostly guests stay for less than 5 days in hotel and for longer stays Resort hotel is preferred. 

● Both hotels have significantly higher booking cancellation rates and very few guests less than 3 % return for another booking in City hotel. 5% guests return for stay in Resort hotel. 

● Most of the guests came from european countries, with most no. of guest coming from Portugal. 

● Guests use different channels for making bookings out of which most preferred way is TA/TO. 

● For hotels higher adr deals come via GDS channel, so hotels should increase their popularity on this channel. 

● Almost 30% of bookings via TA/TO are cancelled. 

● Not getting same room as reserved, longer lead time and waiting time do not affect cancellation of bookings. Although different room allotment do lowers the adr. 

● July- August are the most busier and profitable months for both of hotels. 

● Within a month, adr gradually increases as month ends, with small sudden rise on weekends.
 
● Couples are the most common guests for hotels, hence hotels can plan services according to couples needs to increase revenue. 

● More number of people in guests results in more number of special requests.
 
● Bookings made via complementary market segment and adults have on average high no. of special request. 

● For customers, generally the longer stays (more than 15 days) can result in better deals in terms of low adr.


