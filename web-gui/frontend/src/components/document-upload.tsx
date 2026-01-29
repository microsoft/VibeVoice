"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { ChangeEvent } from "react";
import { Upload, FileText, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface DocumentUploadProps {
  onDocumentLoad: (content: string, filename: string) => void;
  onDocumentChange?: (content: string, filename: string) => void;
  initialContent?: string;
  initialFilename?: string;
}

// Keep the upload limit in a single place so UI and validation stay in sync
const MAX_UPLOAD_BYTES = 1024 * 1024 * 1024; // 1GB
const MAX_UPLOAD_GB = MAX_UPLOAD_BYTES / (1024 * 1024 * 1024);

const isAcceptedTextFile = (file: File) => {
  const lower = file.name.toLowerCase();
  return (
    file.type === "text/markdown" ||
    lower.endsWith(".md") ||
    lower.endsWith(".markdown") ||
    lower.endsWith(".txt")
  );
};

export function DocumentUpload({
  onDocumentLoad,
  onDocumentChange,
  initialContent = "",
  initialFilename = "",
}: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [content, setContent] = useState(initialContent);
  const [filename, setFilename] = useState(initialFilename);
  const [isEditing, setIsEditing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Initialize editor state from initial props on mount only.
    // We intentionally do NOT resync when `initialContent` or `initialFilename` change
    // to avoid clobbering any user edits that occur after mount.
    setContent(initialContent);
    setFilename(initialFilename);
    setIsEditing(Boolean(initialContent || initialFilename));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const handleFileUpload = useCallback(
    async (file: File) => {
      // Enforce maximum size for browser-friendly editing (kept in sync with UI below)
      if (file.size > MAX_UPLOAD_BYTES) {
        toast.error(
          `File too large. Maximum editable size is ${MAX_UPLOAD_GB}GB. For larger files, use chunked upload or a different workflow.`,
        );
        return;
      }

      let text: string;
      try {
        text = await file.text();
      } catch (err) {
        console.error("Error reading file:", err);
        toast.error("Failed to read file contents");
        return;
      }

      setContent(text);
      setFilename(file.name);
      setIsEditing(true);
      onDocumentLoad(text, file.name);
      onDocumentChange?.(text, file.name);
      toast.success(`Document "${file.name}" loaded successfully`);
    },
    [onDocumentLoad, onDocumentChange],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (isAcceptedTextFile(file)) {
          handleFileUpload(file);
        } else {
          toast.error(
            "Please upload a Markdown file (.md, .markdown, or .txt)",
          );
        }
      }
    },
    [handleFileUpload],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (!isAcceptedTextFile(file)) {
          toast.error(
            "Please upload a Markdown file (.md, .markdown, or .txt)",
          );
          return;
        }
        handleFileUpload(file);
      }
    },
    [handleFileUpload],
  );

  const handleClear = useCallback(() => {
    setContent("");
    setFilename("");
    setIsEditing(false);
    onDocumentChange?.("", "");
    toast.info("Document cleared");
  }, [onDocumentChange]);

  const handleStartBlank = useCallback(() => {
    const defaultName = "Untitled.md";
    setContent("");
    setFilename(defaultName);
    setIsEditing(true);
    onDocumentChange?.("", defaultName);
  }, [onDocumentChange]);

  const handleContentChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      setContent(value);
      const nextFilename = filename || "Untitled.md";
      if (!filename) {
        setFilename(nextFilename);
      }
      setIsEditing(true);
      onDocumentChange?.(value, nextFilename);
    },
    [filename, onDocumentChange],
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Document Upload
          </div>
          {filename && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              aria-label="Clear uploaded document"
              title="Clear uploaded document"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!content && !isEditing ? (
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <Upload className="h-12 w-12 text-muted-foreground mb-4 mx-auto" />
            <p className="text-sm text-muted-foreground mb-4">
              Drag and drop your Markdown file here, or
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md,.markdown,.txt"
              onChange={handleFileInput}
              className="hidden"
            />
            <Button
              variant="outline"
              type="button"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Browse Files
            </Button>
            <div className="mt-3">
              <Button variant="ghost" type="button" onClick={handleStartBlank}>
                Start with a blank document
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Supported formats: .md, .markdown, .txt (max {MAX_UPLOAD_GB}GB)
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-muted-foreground">
                <FileText className="h-4 w-4 inline mr-2" />
                {filename}
              </p>
              {/* Header clear button retained for accessibility */}
              {/* NOTE: Duplicate content-area clear button removed — header button remains for accessibility */}
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="markdown-editor">Markdown Editor</Label>
                <textarea
                  id="markdown-editor"
                  value={content}
                  onChange={handleContentChange}
                  placeholder="Start editing your markdown..."
                  className="min-h-[320px] w-full rounded-md border border-input bg-transparent p-3 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                />
              </div>
              <div className="space-y-2">
                <Label>Preview</Label>
                <div className="min-h-[320px] rounded-lg border bg-muted/30 p-4 overflow-auto">
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {content}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
