---
name: web-dev
description: Web application development with React, TypeScript, and modern stacks. Use when building frontend apps, full-stack Next.js projects, or integrating design systems — includes component structure, styling rules, and testing guidance.
---

# Web Development

Standard workflow and conventions for modern web apps.

## Tech Stack Defaults

- **Frontend:** React + TypeScript + Tailwind CSS
- **Backend:** Next.js API routes (or Express when specified)
- **Database:** PostgreSQL
- **Deployment:** Vercel

## Workflow

1. **Design Integration**
   - If a Figma design is available, extract design tokens (colors, typography, spacing) using `TalkToFigma`.
   - Map tokens to Tailwind config or CSS variables.

2. **Development**
   - Structure: `app/`, `components/`, `lib/`, `types/`, `public/`
   - Components: functional with hooks, colocate styles with Tailwind
   - API routes: always include error handling and input validation

3. **Testing**
   - E2E: `playwright` for critical user flows
   - Unit: Jest + React Testing Library for components and utilities

4. **Deployment**
   - Use Vercel for hosting and preview deployments.

## Code Conventions

- ESLint + Prettier are mandatory
- Components: `PascalCase`
- Functions / variables: `camelCase`
- Props: define interfaces in `types/` or next to the component
- Private members: `_prefixWithUnderscore`
