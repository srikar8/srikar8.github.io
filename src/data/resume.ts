export const profile = {
	name: 'Srikar Madarapu',
	title: 'AI Engineer',
	currentRole: 'Forward Deployed Solutions Engineer – Applied AI',
	currentOrg: 'Cisco',
	location: 'Sunnyvale, California',
	email: 'srikarmadarapu24@gmail.com',
	phone: '(470) 738-4285',
	linkedin: 'https://www.linkedin.com/in/srikar8',
	github: 'https://github.com/srikar8',
	resume: '/srikar_madarapu_resume.pdf',
	photo: '/srikar.jpg',
	years: '7+',
	summary:
		"I've spent 7+ years building AI systems that run in production — first computer vision, now LLM agents. I like taking a messy real-world workflow, figuring out which parts actually need a model, and building the whole thing end to end: retrieval, tools, evals, deployment, and the unglamorous reliability work that makes it stick.",
	offDuty: ['Photography', 'Road trips', 'Board & card games'],
};

export const metrics = [
	{ value: 85, suffix: '%', label: 'Manual operations removed', note: 'AI procurement platform' },
	{ value: 60, suffix: '%', label: 'Lower AI workflow cost', note: 'Semantic log filtering' },
	{ value: 40, suffix: '%', label: 'Lower AI workflow latency', note: 'Pattern classification' },
	{ value: 50, suffix: '%', label: 'Faster regression runs', note: '4h → 2h on NX-OS' },
];

export interface GokuForm {
	key: string;
	name: string;
	short: string;
	aura: string;
	img: string;
}

const goku = (key: string, name: string, short: string, aura: string): GokuForm => ({
	key,
	name,
	short,
	aura,
	img: `/sprites/goku/${key}.webp`,
});

export const forms = {
	base: goku('base', 'Base form', 'Base', '#f4f1e8'),
	ssj: goku('ssj', 'Super Saiyan', 'SSJ', '#ffd23f'),
	ssj2: goku('ssj2', 'Super Saiyan 2', 'SSJ2', '#ffe45c'),
	ssj3: goku('ssj3', 'Super Saiyan 3', 'SSJ3', '#ffb81a'),
	ssj4: goku('ssj4', 'Super Saiyan 4', 'SSJ4', '#ff4d5e'),
	ssjb: goku('ssjb', 'Super Saiyan Blue', 'Blue', '#3fd6ff'),
	ultra: goku('ultra', 'Ultra Instinct', 'UI', '#d6dcff'),
};

export interface Upgrade {
	model: string;
	form: GokuForm;
	period: string;
	role: string;
	org: string;
	location: string;
	codename: string;
	changelog: string[];
}

export const upgrades: Upgrade[] = [
	{
		model: '26',
		period: 'Mar 2026 – Present',
		role: 'Forward Deployed Solutions Engineer – Applied AI',
		org: 'Cisco (Contract)',
		location: 'Sunnyvale, CA',
		form: forms.ultra,
		codename: 'Agentic SDLC automation',
		changelog: [
			'Owned discovery, prototyping, and rollout of agentic automation for Nexus NX-OS engineering teams in Data Center Networking.',
			'Built CI investigation and remediation agents correlating Jenkins logs, test results, and code changes — separating product defects, flaky tests, and infra failures, with human-approved fixes.',
			'Reduced AI workflow latency by 40% and cost by 60% with semantic log filtering and pattern classification, reserving LLM reasoning for hard cases.',
			'Cut regression execution time by 50% (4h → 2h) through regression analysis and test workflow optimization.',
			'Built shared agent infrastructure and a unified RAG system powering Webex bots, with citations, metadata filtering, and retrieval evaluation.',
			'Integrated pyATS/Genie checks across NX-OS, IOS XE, and IOS XR to validate device state before and after approved changes.',
			'Deployed and scaled Langfuse and built agent evaluation harnesses; contributed to AI-assisted code review spanning triage, fixes, testing, and merge.',
		],
	},
	{
		model: '25',
		period: 'Jan 2025 – Mar 2026',
		role: 'Full-Stack AI Engineer',
		org: 'Right Pro Inc',
		location: 'Sunnyvale, CA',
		form: forms.ssjb,
		codename: 'Autonomous procurement',
		changelog: [
			'Led end-to-end development of an AI-powered procurement platform, reducing manual operations by 85%.',
			'Architected agentic systems for multistep planning and execution with structured tool calling, persistent memory, secure action schemas, and policy guardrails.',
			'Implemented MCP servers and a decorator-based tool registry for automatic API discovery and agent–tool integration.',
			'Built secure email and document agents with Microsoft Graph API and Azure AI Document Intelligence on a multi-tenant Azure Blob layer.',
			'Shipped via Docker, Azure DevOps, and Argo CD with logging, metrics, and tracing for production SLAs.',
			'Drove 0→1 product development and owned system design, architecture, and production rollout.',
		],
	},
	{
		model: '23',
		period: 'Aug 2023 – Nov 2024',
		role: 'Computer Vision Research Assistant · Graduate TA',
		org: 'MedIX Lab, DePaul University',
		location: 'Chicago, IL',
		form: forms.ssj4,
		codename: 'Multimodal research',
		changelog: [
			'Trained OpenAI CLIP from scratch for semantic image search with FAISS tuned for sub-second retrieval.',
			'Fine-tuned networks for lung-cancer CT multi-classification, improving diagnostic accuracy by 8%.',
			'Cut model training time by 75% with MLflow tracking and hyperparameter optimization.',
			'Compared Vision Transformers vs CNNs for limited medical datasets.',
			'Supported 60+ students in data science, ML, and image processing as a Graduate TA.',
		],
	},
	{
		model: '20',
		period: 'Sep 2019 – Dec 2022',
		role: 'AI & Visual Data Analytics Founder',
		org: 'SRP',
		location: 'Bangalore, India',
		form: forms.ssj3,
		codename: 'Data factory',
		changelog: [
			'Built and grew a visual media startup: 20+ projects with 6+ partner companies across India.',
			'Annotated image/video datasets in CVAT and built ETL pipelines for ML-ready annotations.',
			'Built aerial and 360° image/video data warehouses for AI-driven projects.',
			'Ran analytics and A/B testing that drove a 20% lift in conversions.',
		],
	},
	{
		model: '19',
		period: 'May 2019 – Aug 2019',
		role: 'Video Analytics Engineer',
		org: 'Origa Labs',
		location: 'Bangalore, India',
		form: forms.ssj2,
		codename: 'Real-time vision',
		changelog: [
			'Designed real-time motion detection for human activity monitoring, improving machinery safety alerts by 35%.',
			'Engineered multi-object tracking with Kalman filters and Deep SORT to keep identity through occlusions.',
			'Achieved 3× faster edge inference with OpenVINO quantization and pruning.',
		],
	},
	{
		model: '18',
		period: 'May 2018 – Sep 2018',
		role: 'Computer Vision Engineer',
		org: 'Inferigence Quotient LLP',
		location: 'Bangalore, India',
		form: forms.ssj,
		codename: 'Spatial awareness',
		changelog: [
			'Benchmarked open-source SLAM systems with camera and IMU data.',
			'Improved indoor localization accuracy by 5% and calibrated cameras for 3D mapping.',
		],
	},
	{
		model: '17',
		period: 'Jul 2017 – Dec 2017',
		role: 'Research Intern',
		org: 'DRDO CAIR',
		location: 'Bangalore, India',
		form: forms.base,
		codename: 'First boot',
		changelog: [
			'Implemented CNN-based monocular depth prediction (78% accuracy).',
			'Integrated depth prediction with SLAM for real-time robotic navigation.',
		],
	},
];

export interface Android {
	id: string;
	name: string;
	origin: string;
	mission: string;
	specs: string[];
	hitl: boolean;
	link?: string;
}

export const androids: Android[] = [
	{
		id: '01',
		name: 'CI Investigator',
		origin: 'Cisco',
		mission: "Looks at a failed CI run — Jenkins logs, test results, the diff — and works out whether it's a real bug, a flaky test, or broken infra. Then proposes a fix and waits for a human to approve it.",
		specs: ['LLM reasoning', 'Semantic log filtering', 'Langfuse traces'],
		hitl: true,
	},
	{
		id: '02',
		name: 'Unified RAG Core',
		origin: 'Cisco',
		mission: "One retrieval system behind Webex bots and internal tools. Answers come with citations, and retrieval quality is measured, not guessed.",
		specs: ['RAG', 'Metadata filtering', 'Retrieval evaluation'],
		hitl: false,
	},
	{
		id: '03',
		name: 'Procurement Agent',
		origin: 'Right Pro',
		mission: "Plans and runs multistep procurement tasks on its own. Took about 85% of the manual work out of the process.",
		specs: ['Tool calling', 'Persistent memory', 'Policy guardrails'],
		hitl: true,
	},
	{
		id: '04',
		name: 'Inbox & Docs Agent',
		origin: 'Right Pro',
		mission: "Reads incoming email and documents, pulls out what matters, and files it away — securely, per tenant.",
		specs: ['MS Graph API', 'Azure Doc Intelligence', 'MCP tool registry'],
		hitl: true,
	},
	{
		id: '05',
		name: 'Semantic Seeker',
		origin: 'DePaul MedIX',
		mission: "Search images with text or another image. CLIP trained from scratch, FAISS for sub-second lookup.",
		specs: ['CLIP', 'FAISS', 'PyTorch'],
		hitl: false,
	},
	{
		id: '06',
		name: 'Safety Tracker',
		origin: 'Origa Labs',
		mission: "Tracks people around industrial machines in real time, and doesn't lose them when they walk behind something.",
		specs: ['Deep SORT', 'Kalman filter', 'OpenVINO edge'],
		hitl: false,
	},
];

export const modules = [
	{
		name: 'Agentic & LLM Systems',
		items: ['LLMs', 'RAG', 'LangChain', 'LangGraph', 'LangSmith', 'Langfuse', 'LiteLLM', 'TruLens', 'CrewAI', 'OpenAI Agent Builder', 'ChatKit', 'Prompt Engineering', 'FAISS', 'Pinecone', 'Milvus', 'Temporal', 'Hugging Face', 'Copilot'],
	},
	{
		name: 'Programming & Frameworks',
		items: ['Python', 'TypeScript', 'Node.js', 'FastAPI', 'React', 'PostgreSQL', 'MongoDB', 'Linux', 'REST / GraphQL'],
	},
	{
		name: 'Machine Learning & AI',
		items: ['PyTorch', 'Transformers', 'CLIP', 'Vision Transformers', 'CNNs', 'YOLO', 'Segmentation', 'SLAM', 'Few-shot Learning', 'Transfer Learning', 'Scikit-learn', 'Pandas', 'NumPy', 'JAX'],
	},
	{
		name: 'MLOps & Deployment',
		items: ['Docker', 'Kubernetes', 'CI/CD', 'Jenkins', 'Airflow', 'Argo CD', 'MLflow', 'Weights & Biases', 'Metaflow', 'ONNX', 'Model Optimization'],
	},
	{
		name: 'Cloud & Enterprise',
		items: ['AWS S3', 'SageMaker', 'Lambda', 'EC2', 'Bedrock', 'Kafka', 'Azure AI Foundry', 'Azure Doc Intelligence', 'Azure Blob', 'Snowflake', 'Databricks'],
	},
];

export const education = [
	{ degree: "Master's in Data Science", school: 'DePaul University, Chicago', period: 'Jan 2023 – Nov 2024' },
	{ degree: "Bachelor's in Computer Science", school: 'NIIT University, India', period: 'Jul 2014 – May 2018' },
];

export const certifications = [
	'AWS Certified Generative AI Developer',
	'AWS Certified AI Practitioner',
	'Oracle Cloud Generative AI Professional',
	'AI Code Review — DeepLearning.AI',
	'AI Agents for Image and Video Generation — DeepLearning.AI',
	'Long-Term Agentic Memory with LangGraph',
	'Building Ambient Agents with LangGraph',
];
