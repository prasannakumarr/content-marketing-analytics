## Author performance – traffic, signups, conversion rate
- Goal: See which authors drive volume and convert readers.
- Statistical explanation: Bars for users/signups; conversion line; ranking by weighted conversion rate.
- Analysis/learnings: A few authors dominate traffic; top converters aren’t always the highest-volume authors.
- Recommendations: Double down on high-conversion authors’ playbooks; support traffic-heavy but lower-converting authors with CTA/offer tweaks.
- Plot name: `author_traffic_signup_conversion.png`

## Author performance (log scale)
- Goal: Expose mid-pack authors when traffic is skewed by outliers.
- Statistical explanation: Same as above with log-scaled y-axis to compress heavy tails.
- Analysis/learnings: Reveals mid-tier authors hidden in linear view; conversion differences remain visible.
- Recommendations: Identify mid-pack authors with promising conversion and give them more promotion/testing.
- Plot name: `author_traffic_signup_conversion_log.png`

## Most efficient author (signups per user) with engagement
- Goal: Rank conversion efficiency and add engagement context.
- Statistical explanation: Conversion bars with median line; engagement dots/colorbar and median line on twin axis.
- Analysis/learnings: The top converter stands above median conversion; some authors have strong engagement but only median conversion.
- Recommendations: Replicate top converter’s templates; for high-engagement/median-conversion authors, tune CTAs/offer alignment.
- Plot name: `most_efficient_author.png`

## Traffic vs signups (filtered users ≤ 7000)
- Goal: Understand how traffic maps to signups.
- Statistical explanation: Scatter with regression line on filtered data.
- Analysis/learnings: Positive slope but not steep—traffic helps, but conversion work is still needed.
- Recommendations: Pair acquisition with conversion improvements (landing clarity, CTAs).
- Plot name: `traffic_vs_signup_corr.png`

## Engagement vs bounce by author
- Goal: Assess stickiness by author.
- Statistical explanation: Engagement vs bounce scatter, bubble size = users.
- Analysis/learnings: Authors cluster; some achieve higher engagement with lower bounce, indicating better fit/UX.
- Recommendations: Lift low-engagement/high-bounce authors via targeting, structure, and readability fixes.
- Plot name: `engagement_vs_bounce.png`

## Engagement vs signups & conversion (outliers removed)
- Goal: See if deeper sessions drive signups and conversion.
- Statistical explanation: Regression on filtered data (signups ≤ 140, engagement ≤ 200s); bubbles/color encode users on the signups panel.
- Analysis/learnings: Weak positive trend—more engagement modestly boosts signups and conversion.
- Recommendations: Improve on-page engagement (better intros, internal links) but pair with stronger CTAs; engagement alone isn’t enough.
- Plot name: `engagement_signup_conversion.png`

## Posts needing conversion uplift (users ≤ 6000)
- Goal: Find posts with traffic but weak conversion.
- Statistical explanation: Users vs signups scatter, color = conversion rate; medians shown.
- Analysis/learnings: Points in high-users/low-signups quadrant are prime conversion-fix targets.
- Recommendations: Add/upgrade CTAs, align offers to intent, and test forms on those posts before chasing more traffic.
- Plot name: `posts_needing_conversion_help.png`

## Author quadrant – traffic vs conversion
- Goal: Segment authors by volume vs efficiency.
- Statistical explanation: Medians create four quadrants; color = conversion.
- Analysis/learnings: “High traffic/low conversion” authors need conversion fixes; “high/high” deserve amplification.
- Recommendations: Clone “high/high” playbooks; run CTA/offer experiments for “high traffic/low conversion.”
- Plot name: `quadrant_users_vs_conversion.png`

## Post quadrant – traffic vs conversion (users ≤ 6000)
- Goal: Identify individual posts by volume vs efficiency.
- Statistical explanation: Medians split quadrants; color = conversion; high-traffic/low-conversion zone shaded.
- Analysis/learnings: Highlighted zone marks traffic-rich posts underperforming on conversion.
- Recommendations: Prioritize CRO and intent alignment on shaded-zone posts; then scale winners.
- Plot name: `quadrant_posts_users_vs_conversion.png`

## Correlation – engagement vs conversion (engagement ≤ 170s)
- Goal: Quantify the link between engagement and conversion.
- Statistical explanation: Regression with Pearson r ≈ 0.36 (moderate positive).
- Analysis/learnings: Better engagement modestly lifts conversion.
- Recommendations: Keep boosting engagement, but pair with conversion-specific optimizations.
- Plot name: `correlation_conversion_engagement.png`
- Table: `correlation_conversion_engagement.csv`

## Engagement vs conversion with bounce color (engagement ≤ 200s)
- Goal: See how engagement and conversion move together while visualizing bounce.
- Statistical explanation: Scatter colored by bounce; annotated r for engagement–conversion and engagement–bounce.
- Analysis/learnings: Engagement–conversion is weakly positive; engagement–bounce shows no strong tie—bounce seems driven by other factors.
- Recommendations: Improve engagement and CTAs together; address bounce via intent match, speed, clarity.
- Plot name: `engagement_conversion_bounce_interaction.png`

## Author heatmap – conversion & engagement
- Goal: Quickly spot best and lagging authors on two key quality metrics.
- Statistical explanation: Heatmap with normalized colors, annotated actual values.
- Analysis/learnings: Darker in both columns = best-practice authors; light cells highlight where to coach/iterate.
- Recommendations: Share playbooks from dark/dark authors; coach and test with lighter performers.
- Plot name: `author_heatmap_conversion_engagement.png`

## Quality-Volume-Outcome (engagement↑, bounce↓ → conversion; bubble = traffic)
- Goal: Combine quality signals into one score and see if they predict conversion.
- Statistical explanation: Quality score = engagement z-score minus bounce z-score; regression vs conversion; bubble size/color = users; annotated r.
- Analysis/learnings: Positive correlation—better “quality” (higher engagement, lower bounce) tends to lift conversion, but slope is modest.
- Recommendations: Raise quality (longer, relevant sessions; lower bounce) alongside CRO; focus first on big bubbles with middling conversion.
- Plot name: `quality_volume_outcome.png`

## Bounce + Traffic → Conversion (users ≤ 5000)
- Goal: Test if popular, low-bounce posts convert better.
- Statistical explanation: Traffic vs conversion scatter, color = bounce; regression and annotated correlations.
- Analysis/learnings: Bounce–conversion correlation is moderately negative (~-0.55): lower bounce tends to higher conversion; traffic–conversion is weak.
- Recommendations: Fix high-bounce, high-traffic posts first (intent alignment, CTAs, speed); don’t rely on traffic alone.
- Plot name: `bounce_traffic_conversion.png`
