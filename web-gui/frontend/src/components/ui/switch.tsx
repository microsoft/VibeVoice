"use client"

import * as React from "react"
import * as SwitchPrimitive from "@radix-ui/react-switch"
import { cva } from "class-variance-authority"

import { cn } from "@/lib/utils"

const switchRoot = cva(
  "peer focus-visible:border-ring focus-visible:ring-ring/50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-input dark:data-[state=unchecked]:bg-input/80 group/switch inline-flex shrink-0 items-center rounded-full border border-transparent shadow-xs transition-all outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 [&_svg]:pointer-events-none",
  {
    variants: {
      size: {
        default: "data-[size=default]:h-[1.15rem] data-[size=default]:w-8",
        sm: "data-[size=sm]:h-3.5 data-[size=sm]:w-6",
      },
      state: {
        // state variant left for explicit overrides but background/data-state classes now live in the base
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
)

const switchThumb = cva(
  "bg-background pointer-events-none block rounded-full ring-0 transition-transform data-[state=checked]:translate-x-[calc(100%-2px)] data-[state=unchecked]:translate-x-0",
  {
    variants: {
      size: {
        default: "group-data-[size=default]/switch:size-4",
        sm: "group-data-[size=sm]/switch:size-3",
      },
      state: {
        // translation classes moved into the base so they also apply in uncontrolled mode
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
)

function Switch({
  className,
  size = "default",
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root> & {
  size?: "sm" | "default"
}) {
  // We rely on Radix's `data-state` attributes for visuals; do not derive state manually

  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      data-size={size}
      className={cn(switchRoot({ size }), className)}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className={cn(switchThumb({ size }))}
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
