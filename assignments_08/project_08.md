# Week 8 Project — Supabase and AWS Costs

## Supabase setup

- **Project:** `python200`
- **Project URL:** `https://deetpznrnxfytbxtrvlo.supabase.co`
- **Region:** East US (Ohio), `us-east-2`
- Created the `weather_raw` and `weather_enriched` tables with the required columns. Both tables are empty until the later pipeline assignments load data.
- Row Level Security is disabled on both tables as required for this assignment.
- Supabase API keys are private and are not included here. I will not show a key in the video.

## AWS Pricing Calculator

- **Scenario A — lightweight instance:** `t3.micro`, Linux, 160 hours/month: **$1.66/month**.
- **Scenario B — analytics setup:** `p3.2xlarge` for 730 hours/month (**$2,233.80/month**), Amazon RDS for MySQL `db.m5.large`, single-AZ with 100 GB storage (**$312.81/month**), and 1,024 GB of S3 Standard storage (**$23.55/month**). **Estimated total: $2,570.16/month.** Scenario B is about **$2,568.50/month more** than Scenario A.

## Cost analysis

The GPU instance is the largest cost driver, at about 87% of Scenario B’s estimated monthly total. This shows why runtime matters: a `p3.2xlarge` left on for all 730 hours costs much more than one used only during training. I would stop it when GPU work is finished. The estimates may exclude charges such as data transfer, taxes, and additional storage or backup use.