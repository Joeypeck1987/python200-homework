# Week 8 Warmup

## Part 1: Cloud Concepts

### Question 1: What is the cloud economic model?
Cloud computing rents compute, storage, and networking on demand, so you pay for what you use. Owning servers means paying up front and maintaining enough hardware for peak demand, even when it sits idle.

### Question 2: Vertical and horizontal scaling
Vertical scaling makes one machine bigger (more CPU, RAM, or GPU); horizontal scaling adds machines and splits work. A faster GPU and more RAM for one training job is vertical scaling.

- A web app growing from 1,000 to 100,000 daily users: **horizontal**, by adding app instances to share traffic.
- A slow model-training job needing a faster GPU and more RAM: **vertical**, by upgrading that machine.
- A pipeline growing from 10 to 10,000 splittable files: **horizontal**, by distributing file processing across machines.

### Question 3: Service models
- **Gmail — SaaS:** a finished app; the user signs in and uses it.
- **Azure Virtual Machines — IaaS:** virtual machines that the customer configures and maintains.
- **AWS S3 — IaaS:** cloud storage infrastructure; the customer manages objects, buckets, and access settings.
- **GitHub Codespaces — PaaS:** hosted development environments; GitHub manages the machine while the developer supplies code and configuration.
- **Snowflake — SaaS (managed data platform):** a ready-to-use data warehouse managed by Snowflake.
- **Supabase — BaaS:** a managed backend with PostgreSQL and APIs, so app developers can use backend features without operating database servers.

With **IaaS**, I manage the operating system, software, and much of the security on rented infrastructure (example: Azure Virtual Machines). With **PaaS**, I bring my code and configuration while the provider runs the platform (example: Azure App Service). With **SaaS**, I use a finished product and manage little beyond my account and data (example: Gmail).

### Question 4: Managed data platforms
Databricks or Snowflake package and manage cloud components for data work. They are faster to start with, but can reduce flexibility and add cost or platform dependence. Building directly with AWS or GCP gives more choice and control but takes more setup and expertise.

### Question 5: When local can be better
If the dataset fits comfortably on one computer and the job does not need heavy compute, local work is often simpler and cheaper, especially for a first prototype.

## Part 2: Cloud Landscape

### Question 1: The three hyperscalers
- **AWS:** a broad range of services and a mature ecosystem; common for organizations needing many infrastructure choices.
- **Google Cloud:** strong in data analytics and machine learning; useful for data-heavy and AI-focused organizations.
- **Microsoft Azure:** strong Microsoft product integration; common in organizations already using Microsoft tools and enterprise services.

### Question 2: Why this course moved from Azure to Supabase
- **Access:** students can create Supabase accounts themselves, while Azure organizational invitations and tenant setup could block people.
- **Teaching fit:** Supabase gives us a relational PostgreSQL database, which teaches transferable querying and structured-data skills; Azure Blob Storage mainly stores files.
- **Pipeline fit:** the raw and enriched pipeline stages map clearly to two related Supabase tables that are easy to inspect and debug.

This suggests choosing tools based on access, learning goals, and how well they fit the project rather than choosing only by brand name.

### Question 3: Service categories and examples
- 10 TB of image files retrievable by filename: **object storage**, such as Amazon S3.
- GPU training for four hours, then shut down: **compute**, such as an Amazon EC2 p3.2xlarge GPU instance.
- Web API that scales with traffic: **serverless compute**, such as AWS Lambda.
- Send structured input to an LLM and receive text: **LLM API**, such as Amazon Bedrock or OpenAI.

### Question 4: Example multi-provider stack
For a weather dashboard, I could collect observations with AWS Lambda, store raw and processed records in Supabase PostgreSQL, and ask an OpenAI model to summarize them. Consolidating to one provider can simplify billing, permissions, and networking; I would give up some provider choice and could become more dependent on that provider.
